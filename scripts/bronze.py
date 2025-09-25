#!/usr/bin/env python3
# bronze.py - Couche Bronze : Ingestion des données brutes
# Hackathon EFREI - Ville Durable et Intelligente
# Équipe : Prédiction Inondations Île-de-France

import os
import re
import sys
import requests
import pandas as pd
from urllib.parse import urljoin, urlparse
from requests.adapters import HTTPAdapter, Retry
from pathlib import Path

# === Paramètres ===
DATASET_ID = "6569b51ae64326786e4e8e1a"
DATASET_API = f"https://www.data.gouv.fr/api/1/datasets/{DATASET_ID}/"
HOST = "https://www.data.gouv.fr"
OUT_DIR = Path("data/bronze/meteo")

# Filtres
ALLOWED_PERIODS = {"1950-2023", "2024-2025"}
ALLOWED_BLOCS = {"RR-T-Vent", "autres-parametres"}
ALLOWED_DEPTS = {"75","77","78","91","92","93","94","95"}  # Île-de-France

OUT_DIR.mkdir(parents=True, exist_ok=True)

# === Session HTTP robuste ===
session = requests.Session()
retries = Retry(
    total=5,
    backoff_factor=0.6,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=("GET", "HEAD"),
)
session.mount("https://", HTTPAdapter(max_retries=retries))
session.headers.update({"Accept": "application/json, */*"})

def log(msg: str):
    print(msg, flush=True)

def absolute_url(u: str) -> str:
    if not u:
        return ""
    if u.startswith("/"):
        return urljoin(HOST, u)
    return u

def filename_from_content_disposition(cd: str) -> str | None:
    if not cd:
        return None
    m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd)
    if m:
        return m.group(1)
    return None

def pick_best_hint(res: dict) -> str:
    return (
        (res.get("file") or {}).get("filename")
        or res.get("title")
        or res.get("name")
        or res.get("id")
        or "unknown"
    )

def get_dataset_json():
    log(f"Fetch dataset: {DATASET_API}")
    r = session.get(DATASET_API, timeout=30)
    r.raise_for_status()
    return r.json()

def final_filename_and_url(url: str, hint: str) -> tuple[str, str]:
    url = absolute_url(url)
    h = session.head(url, allow_redirects=True, timeout=30)
    if 200 <= h.status_code < 300:
        cd = h.headers.get("Content-Disposition", "")
        name = filename_from_content_disposition(cd)
        if not name:
            path = urlparse(h.url).path
            name = os.path.basename(path) or hint
        return name, h.url

    g = session.get(url, stream=True, allow_redirects=True, timeout=60)
    g.raise_for_status()
    cd = g.headers.get("Content-Disposition", "")
    name = filename_from_content_disposition(cd)
    if not name:
        path = urlparse(g.url).path
        name = os.path.basename(path) or hint
    g.close()
    return name, g.url

def parse_target(name: str):
    """
    Retourne (code_dept, periode, bloc) si le nom correspond à l'un des formats :
    - QUOT_departement_{dd}_periode_{1950-2023|2024-2025}_{RR-T-Vent|autres-parametres}*.csv.gz
    - QUOTQ_{dd}_{previous|current|latest}-{1950-2023|2024-2025}_{RR-T-Vent|autres-parametres}*.csv.gz
    - Q_{dd}_{previous|current|latest}-{1950-2023|2024-2025}_{RR-T-Vent|autres-parametres}*.csv.gz
    - Q_{dd}_{1950-2023|2024-2025}_{RR-T-Vent|autres-parametres}*.csv.gz
    Sinon None.
    """
    name = os.path.basename(name)
    period_re = r"(1950-2023|2024-2025)"
    bloc_re = r"(RR-T-Vent|autres-parametres)"

    # Canonique
    m = re.match(rf"^QUOT_departement_(\d{{2,3}})_periode_{period_re}_{bloc_re}.*\.csv\.gz$", name, re.I)
    if m:
        return m.group(1), m.group(2), m.group(3)

    # Préfixe QUOTQ_
    m = re.match(rf"^QUOTQ_(\d{{2,3}})_(?:previous|current|latest)-{period_re}_{bloc_re}.*\.csv\.gz$", name, re.I)
    if m:
        return m.group(1), m.group(2), m.group(3)

    # Préfixe Q_ avec previous/current/latest
    m = re.match(rf"^Q_(\d{{2,3}})_(?:previous|current|latest)-{period_re}_{bloc_re}.*\.csv\.gz$", name, re.I)
    if m:
        return m.group(1), m.group(2), m.group(3)

    # Préfixe Q_ simple sans previous/current/latest
    m = re.match(rf"^Q_(\d{{2,3}})_{period_re}_{bloc_re}.*\.csv\.gz$", name, re.I)
    if m:
        return m.group(1), m.group(2), m.group(3)

    return None


def is_allowed(code: str, periode: str, bloc: str) -> bool:
    return (periode in ALLOWED_PERIODS) and (bloc in ALLOWED_BLOCS) and (code in ALLOWED_DEPTS)

def download(url: str, out_path: Path):
    with session.get(url, stream=True, allow_redirects=True, timeout=180) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

def csv_gz_to_parquet_dataset(csv_gz_path: str, parquet_dir: Path):
    """Lit un CSV.gz en streaming et écrit un parquet dataset (dossier de fichiers)"""
    parquet_dir.mkdir(parents=True, exist_ok=True)
    chunks = pd.read_csv(csv_gz_path, compression="gzip", sep=";", low_memory=False, chunksize=200000)
    for i, chunk in enumerate(chunks):
        out_file = parquet_dir / f"part_{i}.parquet"
        chunk.to_parquet(out_file, engine="pyarrow", index=False)
    os.remove(csv_gz_path)
    return parquet_dir

def main():
    try:
        dataset = get_dataset_json()
    except Exception as e:
        log(f"Error dataset: {e}")
        sys.exit(1)

    resources = dataset.get("resources") or []
    log(f"{len(resources)} ressources trouvées")

    converted, skipped = 0, 0

    for i, res in enumerate(resources, 1):
        url = absolute_url(res.get("url", ""))
        if not url:
            continue

        hint = pick_best_hint(res)
        try:
            final_name, final_url = final_filename_and_url(url, hint)
        except Exception:
            continue

        info = parse_target(final_name) or parse_target(hint)
        if not info:
            log(f"Skip {final_name}: ne matche pas parse_target")
            skipped += 1
            continue

        code, periode, bloc = info
        if not is_allowed(code, periode, bloc):
            log(f"Skip {final_name}: code={code}, periode={periode}, bloc={bloc}")
            skipped += 1
            continue

        canonical_csv = f"Q_{code}_{periode}_{bloc}.csv.gz"
        out_csv_path = OUT_DIR / canonical_csv
        out_parquet_dir = OUT_DIR / canonical_csv.replace(".csv.gz", "_parquet")

        if out_parquet_dir.exists():
            log(f"Déjà converti: {out_parquet_dir}")
            continue

        log(f"Téléchargement {canonical_csv} ({periode}, {bloc}, dept {code})")
        try:
            download(final_url, out_csv_path)
            log(f"Téléchargement OK: {out_csv_path}")

            log("Conversion en Parquet dataset...")
            parquet_path = csv_gz_to_parquet_dataset(out_csv_path, out_parquet_dir)
            converted += 1
            log(f"Converti -> {parquet_path}")

        except Exception as e:
            skipped += 1
            log(f"Erreur: {e}")
            if out_csv_path.exists():
                os.remove(out_csv_path)

    log("------------------------------------------------------------")
    log(f"Bilan: convertis={converted}, ignorés={skipped}")
    log(f"Dossier: {OUT_DIR}")
    log("Terminé.")

if __name__ == "__main__":
    main()
