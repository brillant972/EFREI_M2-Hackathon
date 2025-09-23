# gold_etl.py  (robust text decoding)
# - Reads from MariaDB (mf_quot_daily_vent, mf_quot_daily_autres)
# - Writes Gold to Postgres (dim_station, fact_weather_daily, fact_meteo_extras_daily)
# - Handles encodings: try utf8mb4, fallback latin1; also decodes any raw bytes to str

import os
import sys
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

# ------------ ENV / CONNECTIONS ------------
MDB_USER = os.getenv("MDB_USER", "root")
MDB_PASS = os.getenv("MDB_PASS", "1234")
MDB_HOST = os.getenv("MDB_HOST", "127.0.0.1")
MDB_PORT = int(os.getenv("MDB_PORT", "3306"))
MDB_DB   = os.getenv("MDB_NAME", "hackathon")

PG_USER = os.getenv("PG_USER", "postgres")
PG_PASS = os.getenv("PG_PASS", "1234")
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DB   = os.getenv("PG_DB",   "HACKATHON")

SRC_VENT   = "mf_quot_daily_vent"
SRC_AUTRES = "mf_quot_daily_autres"

DIM_STATION             = "dim_station"
FACT_WEATHER_DAILY      = "fact_weather_daily"
FACT_METEO_EXTRAS_DAILY = "fact_meteo_extras_daily"
FACT_FEATURES_DAILY     = "fact_features_daily"
FACT_PREDICTIONS        = "fact_predictions"

CHUNK = 100_000

def make_maria_engine(charset: str) -> Engine:
    uri = f"mysql+pymysql://{MDB_USER}:{MDB_PASS}@{MDB_HOST}:{MDB_PORT}/{MDB_DB}?charset={charset}"
    return create_engine(uri, pool_pre_ping=True, connect_args={"charset": charset})

def make_pg_engine() -> Engine:
    uri = f"postgresql+psycopg2://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}"
    return create_engine(uri, pool_pre_ping=True)

# start with utf8mb4, fallback later if needed
maria_charset = "utf8mb4"
maria: Engine = make_maria_engine(maria_charset)
pg: Engine    = make_pg_engine()

# ------------ HELPERS ------------
def ensure_postgis():
    with pg.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))

def decode_bytes_in_df(df: pd.DataFrame, primary_enc: str) -> pd.DataFrame:
    """
    Ensure all object columns are proper str (unicode). If any cell is bytes,
    decode with primary_enc -> latin1 -> cp1252 (fallback sequence).
    """
    if df.empty:
        return df
    for col in df.columns:
        if df[col].dtype == object:
            def _fix(v):
                if isinstance(v, (bytes, bytearray)):
                    for enc in (primary_enc, "latin1", "cp1252"):
                        try:
                            return v.decode(enc)
                        except Exception:
                            continue
                    # last resort: replace undecodable bytes
                    return v.decode("utf-8", errors="replace")
                return v
            df[col] = df[col].map(_fix)
    return df

def read_sql_maria(query: str, params: dict | None = None) -> pd.DataFrame:
    """
    Read from MariaDB; if we hit a decode error, rebuild engine with latin1 and retry.
    Always normalize any bytes->str to be safe for Postgres.
    """
    global maria, maria_charset
    try:
        df = pd.read_sql(text(query), maria, params=params)
        return decode_bytes_in_df(df, maria_charset)
    except Exception as e:
        msg = str(e).lower()
        # detect decode-ish problems
        if ("codec" in msg and ("utf-8" in msg or "decode" in msg)) or "invalid continuation byte" in msg:
            print("[warn] MariaDB text decode failed with", maria_charset, "-> falling back to latin1")
            try:
                maria.dispose()
            except Exception:
                pass
            maria_charset = "latin1"
            maria = make_maria_engine(maria_charset)
            with maria.begin() as conn:
                conn.execute(text("SET NAMES latin1"))
            df = pd.read_sql(text(query), maria, params=params)
            return decode_bytes_in_df(df, maria_charset)
        raise  # not a text/codec problem — propagate

def create_dim_station():
    print("Building dim_station ...")
    q = f"""
        SELECT NUM_POSTE, NOM_USUEL, LAT, LON, ALTI
        FROM {SRC_VENT}
        UNION
        SELECT NUM_POSTE, NOM_USUEL, LAT, LON, ALTI
        FROM {SRC_AUTRES}
    """
    df = read_sql_maria(q).drop_duplicates(subset=["NUM_POSTE"]).copy()
    df.rename(columns={
        "NUM_POSTE":"station_id",
        "NOM_USUEL":"name",
        "LAT":"lat",
        "LON":"lon",
        "ALTI":"alti"
    }, inplace=True)

    df.to_sql(DIM_STATION, pg, if_exists="replace", index=False)

    with pg.begin() as conn:
        conn.execute(text(f"ALTER TABLE {DIM_STATION} ADD COLUMN IF NOT EXISTS geom geography(Point,4326)"))
        try:
            conn.execute(text(f"ALTER TABLE {DIM_STATION} ADD PRIMARY KEY (station_id)"))
        except Exception:
            pass
        conn.execute(text(f"""
            UPDATE {DIM_STATION}
            SET geom = CASE
                WHEN lon IS NOT NULL AND lat IS NOT NULL
                THEN ST_SetSRID(ST_MakePoint(lon, lat), 4326)::geography
                ELSE NULL
            END
        """))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{DIM_STATION}_geom ON {DIM_STATION} USING GIST(geom)"))

    print(f"dim_station rows: {len(df)}")

def _select_common(df: pd.DataFrame, keep_cols: list) -> pd.DataFrame:
    cols = [c for c in keep_cols if c in df.columns]
    return df.loc[:, cols].copy()

def copy_fact_from_source(src_table: str, dest_table: str, keep_cols: list, pkey=("date","station_id")):
    print(f"Loading {dest_table} from {src_table} ...")
    total = read_sql_maria(f"SELECT COUNT(*) AS n FROM {src_table} WHERE DATE >= '2020-01-01'")["n"].iloc[0]
    if total == 0:
        print(f"{src_table}: 0 rows (DATE >= 2020-01-01), skipping.")
        return

    offset = 0
    wrote_any = False
    while offset < total:
        df = read_sql_maria(f"""
            SELECT * FROM {src_table}
            WHERE DATE >= '2020-01-01'
            ORDER BY DATE, NUM_POSTE
            LIMIT :limit OFFSET :offset
        """, {"limit": CHUNK, "offset": offset})

        if "NUM_POSTE" in df.columns:
            df["station_id"] = df["NUM_POSTE"]
        if "DATE" in df.columns:
            df["date"] = pd.to_datetime(df["DATE"]).dt.date

        df = _select_common(df, keep_cols)
        if df.empty:
            offset += CHUNK
            continue

        if not wrote_any:
            df.to_sql(dest_table, pg, if_exists="replace", index=False)
            wrote_any = True
        else:
            df.to_sql(dest_table, pg, if_exists="append", index=False)

        print(f"  -> {dest_table}: +{len(df)} rows (offset {offset})")
        offset += CHUNK

    with pg.begin() as conn:
        try:
            conn.execute(text(f"ALTER TABLE {dest_table} ADD PRIMARY KEY ({pkey[0]}, {pkey[1]})"))
        except Exception:
            pass
        try:
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{dest_table}_station ON {dest_table}(station_id)"))
        except Exception:
            pass
        try:
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{dest_table}_date ON {dest_table}(date)"))
        except Exception:
            pass

def main():
    ensure_postgis()

    # 1) dim_station
    create_dim_station()

    # 2) fact_weather_daily from VENT
    keep_weather = [
        "date","station_id",
        "RR","QRR","NBRR","RR_ME","RRAB","QRRAB","RRABDAT",
        "NBJRR1","NBJRR5","NBJRR10","NBJRR30","NBJRR50","NBJRR100",
        "TX","QTX","NBTX","TX_ME","TXAB","QTXAB","TXDAT",
        "TN","QTN","NBTN","TN_ME","TNAB","QTNAB","TNDAT",
        "UM","QUM","UN","QUN","UX","QUX","HUN","QHUN","HUX","QHUX",
        "NUM_POSTE","NOM_USUEL","LAT","LON","ALTI"
    ]
    copy_fact_from_source(SRC_VENT, FACT_WEATHER_DAILY, keep_weather)

    # 3) fact_meteo_extras_daily from AUTRES
    keep_extras = [
        "date","station_id",
        "DHUMI40","QDHUMI40","DHUMI80","QDHUMI80",
        "TSVM","QTSVM",
        "ETPMON","QETPMON","ETPGRILLE","QETPGRILLE","ECOULEMENTM","QECOULEMENTM",
        "INST","QINST","GLOT","QGLOT","DIFT","QDIFT","DIRT","QDIRT","INFRART","QINFRART",
        "UV","QUV","UV_INDICEX","QUV_INDICEX","SIGMA","QSIGMA",
        "HNEIGEF","QHNEIGEF","NEIGETOTX","QNEIGETOTX","NEIGETOT06","QNEIGETOT06","NEIG","QNEIG",
        "BROU","QBROU","ORAG","QORAG","GRESIL","QGRESIL","GRELE","QGRELE","ROSEE","QROSEE",
        "VERGLAS","QVERGLAS","SOLNEIGE","QSOLNEIGE","GELEE","QGELEE","FUMEE","QFUMEE",
        "BRUME","QBRUME","ECLAIR","QECLAIR",
        "NB300","QNB300","BA300","QBA300",
        "TMERMIN","QTMERMIN","TMERMAX","QTMERMAX",
        "NUM_POSTE","NOM_USUEL","LAT","LON","ALTI"
    ]
    copy_fact_from_source(SRC_AUTRES, FACT_METEO_EXTRAS_DAILY, keep_extras)

    # 4) optional shells
    with pg.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {FACT_FEATURES_DAILY} (
                date date NOT NULL,
                station_id bigint NOT NULL,
                rr_24h double precision,
                rr_72h double precision,
                rr_7d double precision,
                api_5d double precision,
                api_10d double precision,
                spi_30 double precision,
                spi_90 double precision,
                etp_7d double precision,
                sm40_7d double precision,
                sm80_7d double precision,
                rr_p95_flag boolean,
                PRIMARY KEY (date, station_id)
            )
        """))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {FACT_PREDICTIONS} (
                date date NOT NULL,
                station_id bigint NOT NULL,
                p_flood double precision,
                y_hat double precision,
                model_version text,
                created_at timestamp DEFAULT now(),
                PRIMARY KEY (date, station_id, model_version)
            )
        """))

    print("Gold ETL complete ✅ (MariaDB charset mode:", maria_charset, ")")

if __name__ == "__main__":
    try:
        main()
    except SQLAlchemyError as e:
        print("DB error:", e)
        sys.exit(1)
    except UnicodeDecodeError as e:
        print("Unicode decode error:", e)
        sys.exit(1)
    except Exception as e:
        print("Fatal:", e)
        sys.exit(1)
