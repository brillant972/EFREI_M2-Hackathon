# silver_ingest_split.py (UTF-8 safe)
# - Reads Parquet from bronze/
# - Builds DATE + DATE_STR from AAAAMMJJ
# - Keeps only DATE >= 2020-01-01
# - Writes to silver/
# - Loads into TWO MariaDB tables:
#     * mf_quot_daily_vent              (for *_RR-T-Vent.parquet)
#     * mf_quot_daily_autres            (for *_autres-parametres.parquet)

import os
import re
import time
from glob import glob
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

BRONZE_DIR = "bronze"
SILVER_DIR = "silver"

TABLE_VENT   = "mf_quot_daily_vent"
TABLE_AUTRES = "mf_quot_daily_autres"

# ---- MariaDB (XAMPP typical: root with empty password unless changed) ----
DB_USER = os.getenv("MDB_USER", "root")
DB_PASS = os.getenv("MDB_PASS", "1234")   # <— your current setting
DB_HOST = os.getenv("MDB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("MDB_PORT", "3306"))
DB_NAME = os.getenv("MDB_NAME", "hackathon")

# Force utf8mb4 at URL level
ENGINE_URI = (
    f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    f"?charset=utf8mb4"
)

os.makedirs(SILVER_DIR, exist_ok=True)

def make_engine():
    # Also force utf8mb4 at driver level
    return create_engine(
        ENGINE_URI,
        pool_pre_ping=True,
        pool_recycle=180,
        connect_args={
            "charset": "utf8mb4",
            "use_unicode": True,
            "connect_timeout": 10,
            "read_timeout": 120,
            "write_timeout": 120,
        },
    )

engine = make_engine()

# ---------------- Encoding helpers ----------------

def ensure_session_utf8():
    """Ensure DB, session, and connection speak utf8mb4."""
    with engine.begin() as conn:
        # Make sure the DB exists and is UTF-8 by default
        conn.execute(text(
            f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        ))
        conn.execute(text(f"USE `{DB_NAME}`"))
        # Force the current session to UTF-8 (belt & suspenders)
        conn.execute(text("SET NAMES utf8mb4 COLLATE utf8mb4_unicode_ci"))
        conn.execute(text("SET character_set_client = utf8mb4"))
        conn.execute(text("SET character_set_connection = utf8mb4"))
        conn.execute(text("SET character_set_results = utf8mb4"))

def decode_bytes_in_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure every object column is a proper str (unicode)."""
    if df.empty:
        return df
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]):
            def _fix(v):
                if isinstance(v, (bytes, bytearray)):
                    for enc in ("utf-8", "latin1", "cp1252"):
                        try:
                            return v.decode(enc)
                        except Exception:
                            continue
                    return v.decode("utf-8", errors="replace")
                return v
            df[c] = df[c].map(_fix)
    return df

def alter_table_utf8(table_name: str):
    """Convert table storage to utf8mb4 just in case."""
    with engine.begin() as conn:
        try:
            conn.execute(text(
                f"ALTER TABLE `{table_name}` CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            ))
        except Exception:
            # Table might already be utf8mb4 or not exist yet; ignore.
            pass

# ---------------- Core Helpers ----------------

def detect_bloc_from_filename(path: str) -> str:
    name = os.path.basename(path).lower()
    if "_rr-t-vent" in name:
        return "vent"
    if "_autres-parametres" in name:
        return "autres"
    return ""

def normalize_date_cols(df: pd.DataFrame) -> pd.DataFrame:
    if "AAAAMMJJ" not in df.columns:
        return df.iloc[0:0]
    s = df["AAAAMMJJ"].astype(str).str.extract(r"(\d{8})")[0]
    df["DATE"] = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    df["DATE_STR"] = df["DATE"].dt.strftime("%Y-%m-%d")
    return df

def filter_years_2020_plus(df: pd.DataFrame) -> pd.DataFrame:
    if "DATE" not in df.columns:
        return df.iloc[0:0]
    return df[df["DATE"] >= pd.Timestamp(2020, 1, 1)].copy()

def write_parquet(df: pd.DataFrame, out_path: str):
    df.to_parquet(out_path, engine="pyarrow", index=False)

def create_indexes_if_useful(table_name: str):
    with engine.begin() as conn:
        for stmt in (
            f"CREATE INDEX idx_{table_name}_num  ON `{table_name}` (NUM_POSTE)",
            f"CREATE INDEX idx_{table_name}_date ON `{table_name}` (DATE)",
        ):
            try:
                conn.execute(text(stmt))
            except Exception:
                pass

def load_to_mariadb(df: pd.DataFrame, table_name: str):
    """Insert in small batches with retries; utf8mb4 everywhere."""
    global engine

    # 1) sanitize text (decode any stray bytes)
    df2 = decode_bytes_in_df(df.copy())

    # 2) ensure all object cols are str (not bytes)
    for c in df2.columns:
        if pd.api.types.is_object_dtype(df2[c]):
            df2[c] = df2[c].astype(str)

    # 3) write in smaller chunks to avoid packet/timeouts
    chunk_plan = [1000, 500, 200]
    last_err = None
    first_write = not table_exists(table_name)

    for chunksize in chunk_plan:
        try:
            df2.to_sql(
                name=table_name,
                con=engine,
                if_exists="append",
                index=False,
                chunksize=chunksize,
                method=None,  # keep packets small
            )
            # On the very first creation, enforce utf8mb4 at table-level
            if first_write:
                alter_table_utf8(table_name)
                first_write = False
            return
        except OperationalError as e:
            last_err = e
            try:
                engine.dispose()
            except Exception:
                pass
            engine = make_engine()
            # Re-assert session UTF-8 after reconnect
            ensure_session_utf8()
            time.sleep(1)

    raise last_err if last_err else RuntimeError("Insert failed with unknown error")

def table_exists(table_name: str) -> bool:
    with engine.begin() as conn:
        r = conn.execute(text(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = :db AND table_name = :t"
        ), {"db": DB_NAME, "t": table_name}).scalar()
        return bool(r)

def process_file(path: str):
    bloc = detect_bloc_from_filename(path)
    if bloc == "":
        print(f"[skip] {os.path.basename(path)} -> bloc non détecté (ni VENT ni AUTRES)")
        return

    df = pd.read_parquet(path)
    df = normalize_date_cols(df)
    df = filter_years_2020_plus(df)
    if df.empty:
        print(f"[skip] {os.path.basename(path)} -> 0 lignes 2020+")
        return

    out_name = os.path.basename(path)
    out_path = os.path.join(SILVER_DIR, out_name)
    write_parquet(df, out_path)
    print(f"[silver] écrit: {out_path}  (rows={len(df)})")

    table = TABLE_VENT if bloc == "vent" else TABLE_AUTRES
    load_to_mariadb(df, table)
    print(f"[mariadb] table={table}  rows+={len(df)})")

# ---------------- Main ----------------

def main():
    # Make sure DB/session are utf8mb4
    ensure_session_utf8()

    files = sorted(glob(os.path.join(BRONZE_DIR, "QUOT_departement_*_periode_*.parquet")))
    if not files:
        print("Aucun fichier Parquet trouvé dans bronze/.")
        return

    loaded_tables = set()

    for f in files:
        try:
            process_file(f)
            if "_rr-t-vent" in f.lower():
                loaded_tables.add(TABLE_VENT)
            elif "_autres-parametres" in f.lower():
                loaded_tables.add(TABLE_AUTRES)
        except Exception as e:
            print(f"[error] {os.path.basename(f)}: {e}")

    for t in loaded_tables:
        try:
            create_indexes_if_useful(t)
        except Exception as e:
            print(f"[index warn] {t}: {e}")

if __name__ == "__main__":
    main()
