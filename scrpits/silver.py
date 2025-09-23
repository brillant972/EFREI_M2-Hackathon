# silver_ingest.py
# Transforme les fichiers QUOT* du dossier bronze/ vers silver/ :
# - convertit AAAAMMJJ -> colonnes DATE (date) et DATE_STR ("YYYY-MM-DD")
# - garde uniquement AAAAMMJJ >= 20200101 (i.e. années 2020+)
# - écrit en Parquet dans silver/
# - charge aussi les données dans MariaDB (tables InnoDB) en append

import os
import re
from glob import glob
import pandas as pd
from sqlalchemy import create_engine, text

BRONZE_DIR = "bronze"
SILVER_DIR = "silver"

# ---- Config MariaDB (mettez vos paramètres / variables d'env) ----
DB_USER = os.getenv("MDB_USER", "root")
DB_PASS = os.getenv("MDB_PASS", "password")
DB_HOST = os.getenv("MDB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("MDB_PORT", "3306"))
DB_NAME = os.getenv("MDB_NAME", "meteodata")

# Exemple de driver : mysqlclient, PyMySQL, mariadb-connector-python
# Choisissez celui que vous avez installé :
ENGINE_URI = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"

os.makedirs(SILVER_DIR, exist_ok=True)

engine = create_engine(ENGINE_URI, pool_pre_ping=True)

def table_name_from_file(path: str) -> str:
    """Construit un nom de table à partir du nom de fichier QUOT_departement_.._...parquet."""
    base = os.path.basename(path)
    base = re.sub(r"\.parquet$", "", base, flags=re.I)
    # MariaDB: lettres/chiffres/_ seulement
    base = re.sub(r"[^0-9a-zA-Z_]", "_", base)
    # limite raisonnable
    return base.lower()[:64]

def normalize_date_cols(df: pd.DataFrame) -> pd.DataFrame:
    if "AAAAMMJJ" not in df.columns:
        return df
    # AAAAMMJJ attendu comme entier/chaine "YYYYMMDD"
    # On force en str sur 8 car. puis parse
    s = df["AAAAMMJJ"].astype(str).str.extract(r"(\d{8})")[0]
    df["DATE"] = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    df["DATE_STR"] = df["DATE"].dt.strftime("%Y-%m-%d")  # <-- format YYYY-MM-DD
    return df

def filter_years_2020_plus(df: pd.DataFrame) -> pd.DataFrame:
    if "DATE" not in df.columns:
        return df.iloc[0:0]  # vide si pas de date
    return df[df["DATE"] >= pd.Timestamp(2020, 1, 1)].copy()

def write_parquet(df: pd.DataFrame, out_path: str):
    # Utilise pyarrow si dispo
    df.to_parquet(out_path, engine="pyarrow", index=False)

def load_to_mariadb(df: pd.DataFrame, table: str):
    # Optionnel : création BDD si besoin
    with engine.begin() as conn:
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}`"))
        conn.execute(text(f"USE `{DB_NAME}`"))

    # Ecriture en append (crée la table si elle n'existe pas)
    # Astuce : pour limiter la taille des textes, convertissons object->str raisonnable
    df2 = df.copy()
    for c in df2.columns:
        if pd.api.types.is_object_dtype(df2[c]):
            df2[c] = df2[c].astype(str)

    df2.to_sql(
        name=table,
        con=engine,
        if_exists="append",
        index=False,
        chunksize=5000,
        method="multi",
    )

    # Index utiles (créés une seule fois)
    with engine.begin() as conn:
        cols = df.columns
        if "NUM_POSTE" in cols:
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{table}_num ON `{table}` (NUM_POSTE)"))
        if "DATE" in cols:
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{table}_date ON `{table}` (DATE)"))

def process_file(path: str):
    # On s'attend à des fichiers Parquet déjà en bronze
    df = pd.read_parquet(path)
    df = normalize_date_cols(df)
    df = filter_years_2020_plus(df)
    if df.empty:
        print(f"[skip] {os.path.basename(path)} -> aucun enregistrement 2020+")
        return

    # On garde AAAAMMJJ pour traçabilité, mais on a aussi DATE/DATE_STR
    out_name = os.path.basename(path)  # même nom
    out_path = os.path.join(SILVER_DIR, out_name)

    write_parquet(df, out_path)
    print(f"[silver] écrit: {out_path}  (rows={len(df)})")

    table = table_name_from_file(out_name)
    load_to_mariadb(df, table)
    print(f"[mariadb] table={table}  rows+={len(df)}")

def main():
    files = sorted(glob(os.path.join(BRONZE_DIR, "QUOT_departement_*_periode_*.parquet")))
    if not files:
        print("Aucun fichier Parquet trouvé dans bronze/.")
        return
    for f in files:
        try:
            process_file(f)
        except Exception as e:
            print(f"[error] {os.path.basename(f)}: {e}")

if __name__ == "__main__":
    main()
