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
import gc

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

    # Colonnes cibles (on ne lit que celles-ci si dispo)
    KEEP_BASE = [
        "NUM_POSTE", "NOM_USUEL", "LAT", "LON", "ALTI",
        "date", "AAAAMMJJ",
        "TN", "TX", "TM", "RR", "PMER", "FFM"
    ]

    def __init__(self):
        self.bronze_path = Path("data/bronze/meteo")
        self.silver_path = Path("data/silver")
        self.time_series_path = self.silver_path / "time_series"
        self.features_path = self.silver_path / "features"
        for p in [self.silver_path, self.time_series_path, self.features_path]:
            p.mkdir(parents=True, exist_ok=True)

        # Seuils de nettoyage
        self.max_missing_ratio = 0.95
        self.key_cols_always_keep = {"NUM_POSTE", "NOM_USUEL", "date", "LAT", "LON", "ALTI"}
        self.row_key_any_of = ["RR", "TM", "TN", "TX"]

    # ---------- Utils d’IO mémoire ----------
    def _read_one_file(self, f: Path) -> pd.DataFrame:
        """Lit un fichier en NE CHARGEANT QUE les colonnes utiles. Convertit les types au passage."""
        # Pour Parquet, on peut demander les colonnes directement
        if f.suffix == ".parquet":
            # On ne demande que celles possibles (intersection)
            # Lecture des metadonnées pour savoir ce qui existe (optionnel, sinon on tente direct)
            try:
                df = pd.read_parquet(f, columns=self.KEEP_BASE)
            except Exception:
                # Si échec (colonnes manquantes), on lit tout et on filtrera
                df = pd.read_parquet(f)
        else:
            # CSV.gz
            try:
                df = pd.read_csv(
                    f,
                    compression="gzip",
                    sep=";",
                    low_memory=False,
                    usecols=lambda c: c in set(self.KEEP_BASE)
                )
            except Exception:
                # fallback si le usecols échoue
                df = pd.read_csv(f, compression="gzip", sep=";", low_memory=False)

        # On garde UNIQUEMENT les colonnes d’intérêt ou AAAAMMJJ (pour reconstruire date)
        cols_present = [c for c in self.KEEP_BASE if c in df.columns]
        df = df[[c for c in cols_present]].copy()

        # Normalisation date
        if "date" not in df.columns:
            if "AAAAMMJJ" in df.columns:
                df["date"] = pd.to_datetime(df["AAAAMMJJ"].astype(str), format="%Y%m%d", errors="coerce")
                df.drop(columns=["AAAAMMJJ"], inplace=True, errors="ignore")
            else:
                df["date"] = pd.NaT
        else:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # NUM_POSTE en string pour préserver les zéros en tête
        if "NUM_POSTE" in df.columns:
            df["NUM_POSTE"] = df["NUM_POSTE"].astype(str)

        # Numeric downcast (réduit la RAM)
        for c in ["LAT", "LON", "ALTI", "TN", "TM", "TX", "RR", "PMER", "FFM"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
                if pd.api.types.is_float_dtype(df[c]):
                    df[c] = df[c].astype("float32")
                elif pd.api.types.is_integer_dtype(df[c]):
                    df[c] = df[c].astype("Int32")

        # NOM_USUEL en catégorie (gain RAM)
        if "NOM_USUEL" in df.columns and df["NOM_USUEL"].dtype != "category":
            df["NOM_USUEL"] = df["NOM_USUEL"].astype("category")

        return df

    # 1) Chargement Bronze (récursif) — optimisé
    def load_bronze_data(self) -> pd.DataFrame:
        if not self.bronze_path.exists():
            raise FileNotFoundError(f"Chemin introuvable: {self.bronze_path.resolve()}")

        parquet_files = list(self.bronze_path.rglob("*.parquet"))
        csv_gz_files = [] if parquet_files else list(self.bronze_path.rglob("*.csv.gz"))
        files = parquet_files if parquet_files else csv_gz_files
        if not files:
            raise FileNotFoundError(f"Aucun .parquet ni .csv.gz trouvé sous {self.bronze_path.resolve()}")

        logger.info(f"Fichiers détectés: {len(files)}")

        parts = []
        for i, f in enumerate(files, 1):
            try:
                df = self._read_one_file(f)
                parts.append(df)
            except Exception as e:
                logger.warning(f"Lecture impossible pour {f}: {e}")
                continue

            # Concaténation par paquets pour limiter le pic mémoire
            if len(parts) >= 8:
                parts = [pd.concat(parts, ignore_index=True, copy=False)]
                gc.collect()

        meteo_df = pd.concat(parts, ignore_index=True, copy=False)
        logger.info(f"Météo concaténée: {meteo_df.shape[0]} lignes, {meteo_df.shape[1]} colonnes")
        return meteo_df

    # 2) Nettoyage
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Dédupes
        n0 = len(df)
        if {"NUM_POSTE", "date"}.issubset(df.columns):
            df = df.drop_duplicates(subset=["NUM_POSTE", "date"])
        else:
            df = df.drop_duplicates()
        logger.info(f"Doublons supprimés: {n0 - len(df)}")

        # Bornes plausibles
        if "TN" in df:   df.loc[(df["TN"] < -25) | (df["TN"] > 45), "TN"] = np.nan
        if "TX" in df:   df.loc[(df["TX"] < -20) | (df["TX"] > 55), "TX"] = np.nan
        if "TM" in df:   df.loc[(df["TM"] < -20) | (df["TM"] > 50), "TM"] = np.nan
        if "RR" in df:   df.loc[(df["RR"] < 0)  | (df["RR"] > 300), "RR"] = np.nan
        if "PMER" in df: df.loc[(df["PMER"] < 870) | (df["PMER"] > 1080), "PMER"] = np.nan
        if "FFM" in df:  df.loc[(df["FFM"] < 0)  | (df["FFM"] > 60), "FFM"] = np.nan

        # Trie + interpolation courte par station
        sort_cols = [c for c in ["NUM_POSTE", "date"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols)

        for col in ["TN", "TX", "TM", "RR", "PMER", "FFM"]:
            if col in df.columns and "NUM_POSTE" in df.columns:
                s = df.groupby("NUM_POSTE")[col].apply(lambda x: x.interpolate(limit=3))
                df[col] = s.reset_index(level=0, drop=True).astype(df[col].dtype)

        # Drop colonnes quasi vides / constantes (hors clés protégées)
        missing_ratio = df.isna().mean()
        to_drop_missing = [
            c for c in df.columns
            if (missing_ratio.get(c, 0.0) > self.max_missing_ratio) and (c not in self.key_cols_always_keep)
        ]
        nunique = df.nunique(dropna=True)
        to_drop_constant = [
            c for c in df.columns
            if (nunique.get(c, 0) <= 1) and (c not in self.key_cols_always_keep)
        ]
        to_drop = sorted(set(to_drop_missing) | set(to_drop_constant))
        if to_drop:
            logger.info(f"Colonnes supprimées (quasi vides/constantes): {to_drop}")
            df = df.drop(columns=to_drop, errors="ignore")

        # Drop lignes sans info météo clé
        exist_keys = [c for c in self.row_key_any_of if c in df.columns]
        dropped_rows = 0
        if exist_keys:
            mask_all_nan = df[exist_keys].isna().all(axis=1)
            dropped_rows = int(mask_all_nan.sum())
            if dropped_rows > 0:
                df = df.loc[~mask_all_nan].copy()
        logger.info(f"Lignes supprimées sans info météo ({'+'.join(exist_keys)}): {dropped_rows}")

        # Rapport
        cleaning_report = {
            "input_rows": int(n0),
            "output_rows": int(len(df)),
            "dropped_rows_no_meteo_info": dropped_rows,
            "dropped_columns_missing_ratio_gt": self.max_missing_ratio,
            "dropped_columns_missing": to_drop_missing,
            "dropped_columns_constant": to_drop_constant,
            "remaining_columns": list(df.columns),
            "created_at": datetime.now().isoformat()
        }
        with open(self.features_path / "cleaning_report.json", "w", encoding="utf-8") as f:
            json.dump(cleaning_report, f, indent=2, ensure_ascii=False)

        logger.info("Nettoyage terminé.")
        gc.collect()
        return df

    # 3) Features
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        sort_cols = [c for c in ["NUM_POSTE", "date"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols).copy()

        out = []
        groups = df.groupby("NUM_POSTE", sort=False) if "NUM_POSTE" in df.columns else [(None, df)]

        for _, s in groups:
            s = s.copy()

            if "RR" in s.columns:
                s["RR_7d"]  = s["RR"].rolling(7, min_periods=1).sum().astype("float32")
                s["RR_14d"] = s["RR"].rolling(14, min_periods=1).sum().astype("float32")
                s["RR_30d"] = s["RR"].rolling(30, min_periods=1).sum().astype("float32")

                # API
                api = 0.0
                api_vals = []
                k = 0.9
                rr_series = s["RR"].fillna(0).astype("float32")
                for v in rr_series:
                    api = api * k + float(v)
                    api_vals.append(api)
                s["API"] = np.array(api_vals, dtype="float32")

            if "TM" in s.columns:
                s["TM_7d"]  = s["TM"].rolling(7, min_periods=1).mean().astype("float32")
                s["TM_30d"] = s["TM"].rolling(30, min_periods=1).mean().astype("float32")

            out.append(s)

        enriched = pd.concat(out, ignore_index=True, copy=False)

        # Calendrier
        if "date" not in enriched.columns:
            raise KeyError("La colonne 'date' est manquante après nettoyage.")
        enriched["year"]        = enriched["date"].dt.year
        enriched["month"]       = enriched["date"].dt.month
        enriched["day_of_year"] = enriched["date"].dt.dayofyear
        enriched["sin_doy"]     = np.sin(2 * np.pi * enriched["day_of_year"] / 365.25).astype("float32")
        enriched["cos_doy"]     = np.cos(2 * np.pi * enriched["day_of_year"] / 365.25).astype("float32")

        logger.info("Features créées.")
        gc.collect()
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
        print(f"Rapport nettoyage : {s.features_path / 'cleaning_report.json'}")
    except Exception as e:
        logger.error(f"Erreur Silver Layer : {e}")
        raise

if __name__ == "__main__":
    main()
