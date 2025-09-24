#!/usr/bin/env python3
# silver.py - Couche Silver : Nettoyage et feature engineering (météo uniquement)

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
import warnings
import os
import json

os.makedirs("logs", exist_ok=True)
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/silver_processing.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SilverDataProcessing:
    """Couche Silver : Nettoyage, enrichissement et feature engineering météo"""

    def __init__(self):
        # Chemins (racine du repo -> data/bronze/meteo)
        self.bronze_path = Path("data/bronze/meteo")
        self.silver_path = Path("data/silver")
        self.time_series_path = self.silver_path / "time_series"
        self.features_path = self.silver_path / "features"

        for p in [self.silver_path, self.time_series_path, self.features_path]:
            p.mkdir(parents=True, exist_ok=True)

    # 1) Chargement Bronze (récursif)
    def load_bronze_data(self) -> pd.DataFrame:
        """Charge et concatène tous les fichiers météo (parquet de façon récursive, sinon csv.gz)."""
        if not self.bronze_path.exists():
            raise FileNotFoundError(f"Chemin introuvable: {self.bronze_path.resolve()}")

        parquet_files = list(self.bronze_path.rglob("*.parquet"))
        csv_gz_files = [] if parquet_files else list(self.bronze_path.rglob("*.csv.gz"))

        files = parquet_files if parquet_files else csv_gz_files
        if not files:
            raise FileNotFoundError(f"Aucun .parquet ni .csv.gz trouvé sous {self.bronze_path.resolve()}")

        logger.info(f"Fichiers détectés: {len(files)}")
        dfs = []
        for f in files:
            try:
                if f.suffix == ".parquet":
                    df = pd.read_parquet(f)
                else:
                    df = pd.read_csv(f, compression="gzip", sep=";", low_memory=False)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Lecture impossible pour {f}: {e}")

        if not dfs:
            raise RuntimeError("Aucun fichier lisible.")

        meteo_df = pd.concat(dfs, ignore_index=True)

        # Normalise la date
        if "date" not in meteo_df.columns:
            if "AAAAMMJJ" in meteo_df.columns:
                meteo_df["date"] = pd.to_datetime(meteo_df["AAAAMMJJ"].astype(str), format="%Y%m%d", errors="coerce")
            else:
                raise KeyError("Aucune colonne date ou AAAAMMJJ trouvée.")
        else:
            meteo_df["date"] = pd.to_datetime(meteo_df["date"], errors="coerce")

        # Harmonise les types géographiques si présents
        for c in ["LAT", "LON", "ALTI"]:
            if c in meteo_df.columns:
                meteo_df[c] = pd.to_numeric(meteo_df[c], errors="coerce")

        logger.info(f"Météo concaténée: {meteo_df.shape[0]} lignes, {meteo_df.shape[1]} colonnes")
        return meteo_df

    # 2) Nettoyage
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoie les données météo brutes et garde les colonnes utiles."""
        keep_cols = [
            "NUM_POSTE", "NOM_USUEL", "LAT", "LON", "ALTI", "date",
            "TN", "TX", "TM", "RR", "PMER", "FFM"
        ]
        cols = [c for c in keep_cols if c in df.columns]
        df = df[cols].copy()

        # Dédupes
        n0 = len(df)
        df = df.drop_duplicates(subset=["NUM_POSTE", "date"])
        logger.info(f"Doublons supprimés: {n0 - len(df)}")

        # Bornes plausibles
        if "TN" in df:   df.loc[(df["TN"] < -25) | (df["TN"] > 45), "TN"] = np.nan
        if "TX" in df:   df.loc[(df["TX"] < -20) | (df["TX"] > 55), "TX"] = np.nan
        if "TM" in df:   df.loc[(df["TM"] < -20) | (df["TM"] > 50), "TM"] = np.nan
        if "RR" in df:   df.loc[(df["RR"] < 0) | (df["RR"] > 300),  "RR"] = np.nan
        if "PMER" in df: df.loc[(df["PMER"] < 870) | (df["PMER"] > 1080), "PMER"] = np.nan
        if "FFM" in df:  df.loc[(df["FFM"] < 0) | (df["FFM"] > 60), "FFM"] = np.nan

        # Trie pour interpolation
        df = df.sort_values(["NUM_POSTE", "date"])

        # Interpolation locale par station (petits trous)
        for col in ["TN", "TX", "TM", "RR", "PMER", "FFM"]:
            if col in df.columns:
                df[col] = (
                    df.groupby("NUM_POSTE")[col]
                      .apply(lambda s: s.interpolate(limit=3))
                      .reset_index(level=0, drop=True)
                )

        logger.info("Nettoyage terminé.")
        return df

    # 3) Features
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crée des variables dérivées simples utiles pour l'analyse/ML météo-only."""
        df = df.sort_values(["NUM_POSTE", "date"]).copy()
        out = []

        for station, s in df.groupby("NUM_POSTE", sort=False):
            s = s.copy()

            # Cumuls pluie (si RR présent)
            if "RR" in s:
                s["RR_7d"]  = s["RR"].rolling(7, min_periods=1).sum()
                s["RR_14d"] = s["RR"].rolling(14, min_periods=1).sum()
                s["RR_30d"] = s["RR"].rolling(30, min_periods=1).sum()

                # API
                api = 0.0
                api_vals = []
                k = 0.9
                rr_series = s["RR"].fillna(0)
                for v in rr_series:
                    api = api * k + float(v)
                    api_vals.append(api)
                s["API"] = api_vals

            # Moyennes glissantes TM
            if "TM" in s:
                s["TM_7d"]  = s["TM"].rolling(7, min_periods=1).mean()
                s["TM_30d"] = s["TM"].rolling(30, min_periods=1).mean()

            out.append(s)

        enriched = pd.concat(out, ignore_index=True)

        # Features calendaires
        enriched["year"]        = enriched["date"].dt.year
        enriched["month"]       = enriched["date"].dt.month
        enriched["day_of_year"] = enriched["date"].dt.dayofyear
        enriched["sin_doy"]     = np.sin(2 * np.pi * enriched["day_of_year"] / 365.25)
        enriched["cos_doy"]     = np.cos(2 * np.pi * enriched["day_of_year"] / 365.25)

        logger.info("Features créées.")
        return enriched

    # 4) Sauvegarde
    def save(self, df: pd.DataFrame):
        out_file = self.time_series_path / "meteo_clean.parquet"
        df.to_parquet(out_file, index=False)

        meta = {
            "features": list(df.columns),
            "nb_lignes": int(len(df)),
            "created_at": datetime.now().isoformat()
        }
        with open(self.features_path / "features_summary.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info(f"Sauvegardé: {out_file}")
        return out_file

def main():
    print("SILVER LAYER - Nettoyage et Feature Engineering (Météo)")
    print("=" * 60)
    s = SilverDataProcessing()
    try:
        raw = s.load_bronze_data()
        clean = s.clean_data(raw)
        enriched = s.create_features(clean)
        out_path = s.save(enriched)

        print("\nSILVER LAYER TERMINÉE !")
        print("=" * 60)
        print(f"Lignes finales : {len(enriched)}")
        print(f"Colonnes       : {len(enriched.columns)}")
        print(f"Fichier : {out_path}")
    except Exception as e:
        logger.error(f"Erreur Silver Layer : {e}")
        raise

if __name__ == "__main__":
    main()
