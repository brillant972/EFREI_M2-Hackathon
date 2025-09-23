import os
import re
import sys
import requests
import pandas as pd
from urllib.parse import urljoin, urlparse
from requests.adapters import HTTPAdapter, Retry

# === Paramètres ===
DATASET_ID = "6569b51ae64326786e4e8e1a"
DATASET_API = f"https://www.data.gouv.fr/api/1/datasets/{DATASET_ID}/"
HOST = "https://www.data.gouv.fr"
OUT_DIR = "bronze"

# Cibles demandées
ALLOWED_PERIODS = {"1950-2023", "2024-2025"}
ALLOWED_BLOCS = {"RR-T-Vent", "autres-parametres"}

# === Dossier de sortie ===
os.makedirs(OUT_DIR, exist_ok=True)

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
    log(f"📥 Fetch dataset: {DATASET_API}")
    r = session.get(DATASET_API, timeout=30)
    r.raise_for_status()
    return r.json()

def final_filename_and_url(url: str, hint: str) -> tuple[str, str]:
    """Suit les redirections et récupère le nom final sans télécharger le contenu."""
    url = absolute_url(url)

    # HEAD d'abord
    try:
        h = session.head(url, allow_redirects=True, timeout=30)
        if 200 <= h.status_code < 300:
            cd = h.headers.get("Content-Disposition", "")
            name = filename_from_content_disposition(cd)
            if not name:
                path = urlparse(h.url).path
                name = os.path.basename(path) or hint
            return name, h.url
    except requests.RequestException:
        pass

    # Fallback GET stream (on ne lit pas le corps)
    g = session.get(url, stream=True, allow_redirects=True, timeout=60)
    g.raise_for_status()
    cd = g.headers.get("Content-Disposition", "")
    name = filename_from_content_disposition(cd)
    if not name:
        path = urlparse(g.url).path
        name = os.path.basename(path) or hint
    g.close()
    return name, g.url

# --- Parseur des formats de noms acceptés ---
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

    # Préfixe QUOTQ_ avec previous/current/latest
    m = re.match(rf"^QUOTQ_(\d{{2,3}})_(?:previous|current|latest)-{period_re}_{bloc_re}.*\.csv\.gz$", name, re.I)
    if m:
        return m.group(1), m.group(2), m.group(3)

    # Préfixe Q_ avec previous/current/latest (ex: Q_19_previous-1950-2023_autres-parametres.csv.gz)
    m = re.match(rf"^Q_(\d{{2,3}})_(?:previous|current|latest)-{period_re}_{bloc_re}.*\.csv\.gz$", name, re.I)
    if m:
        return m.group(1), m.group(2), m.group(3)

    # Préfixe Q_ simple sans previous/current/latest
    m = re.match(rf"^Q_(\d{{2,3}})_{period_re}_{bloc_re}.*\.csv\.gz$", name, re.I)
    if m:
        return m.group(1), m.group(2), m.group(3)

    return None

def is_allowed(periode: str, bloc: str) -> bool:
    return (periode in ALLOWED_PERIODS) and (bloc in ALLOWED_BLOCS)

def download(url: str, out_path: str):
    with session.get(url, stream=True, allow_redirects=True, timeout=180) as r:
        r.raise_for_status()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

def csv_gz_to_parquet(csv_gz_path: str) -> str:
    """Lit un CSV.gz (séparateur ;) et écrit un .parquet à côté. Supprime le CSV.gz si OK."""
    parquet_path = csv_gz_path.replace(".csv.gz", ".parquet")
    df = pd.read_csv(csv_gz_path, compression="gzip", sep=";", low_memory=False)
    df.to_parquet(parquet_path, engine="pyarrow", index=False)
    os.remove(csv_gz_path)
    return parquet_path

def main():
    try:
        dataset = get_dataset_json()
    except requests.HTTPError as e:
        log(f"❌ HTTP error dataset: {e}")
        sys.exit(1)
    except Exception as e:
        log(f"❌ Error dataset: {e}")
        sys.exit(1)

    resources = dataset.get("resources") or []
    log(f"🔎 {len(resources)} ressources trouvées")

    downloaded, converted, skipped = 0, 0, 0

    for i, res in enumerate(resources, 1):
        url = absolute_url(res.get("url", ""))
        if not url:
            skipped += 1
            log(f"[{i}] ⏭️  Pas d'URL, skip")
            continue

        hint = pick_best_hint(res)
        log(f"[{i}] Ressource: hint='{hint}' | url={url}")

        # Nom final après redirections
        try:
            final_name, final_url = final_filename_and_url(url, hint)
        except requests.HTTPError as e:
            skipped += 1
            log(f"    ❌ HEAD/GET metadata error: {e}")
            continue
        except Exception as e:
            skipped += 1
            log(f"    ❌ Metadata error: {e}")
            continue

        log(f"    → Nom final détecté: {final_name}")

        info = parse_target(final_name) or parse_target(hint)
        if not info:
            skipped += 1
            log("    🚫 Ne matche pas les formats quotidiens attendus, skip")
            continue

        code, periode, bloc = info

        # Filtrage strict selon la demande (périodes 1950-2023 et 2024-2025 + deux blocs)
        if not is_allowed(periode, bloc):
            skipped += 1
            log(f"    🚫 Filtré (periode={periode}, bloc={bloc})")
            continue

        # Nom canonique demandé
        canonical_csv = f"QUOT_departement_{code}_periode_{periode}_{bloc}.csv.gz"
        out_csv_path = os.path.join(OUT_DIR, canonical_csv)

        if os.path.exists(out_csv_path.replace(".csv.gz", ".parquet")):
            log(f"    ✅ Parquet déjà présent: {os.path.basename(out_csv_path).replace('.csv.gz','.parquet')} (skip)")
            continue

        log(f"    ⬇️  Téléchargement → {canonical_csv}")
        try:
            download(final_url, out_csv_path)

            if os.path.exists(out_csv_path):
                size_mb = os.path.getsize(out_csv_path) / 1_000_000
                downloaded += 1
                log(f"    ✅ OK ({size_mb:.2f} MB)")

                try:
                    parquet_path = csv_gz_to_parquet(out_csv_path)
                    converted += 1
                    log(f"    💾 Converti en Parquet: {parquet_path}")
                except Exception as e:
                    log(f"    ❌ Erreur conversion Parquet (CSV conservé): {e}")
            else:
                skipped += 1
                log("    ❌ Téléchargement annoncé mais fichier introuvable")

        except Exception as e:
            if os.path.exists(out_csv_path):
                try:
                    os.remove(out_csv_path)
                except OSError:
                    pass
            skipped += 1
            log(f"    ❌ Erreur téléchargement: {e}")

    log("—" * 60)
    log(f"📊 Bilan: téléchargés={downloaded}, convertis_parquet={converted}, ignorés={skipped}")
    log(f"📂 Dossier: ./{OUT_DIR}")
    log("🎉 Terminé.")

if __name__ == "__main__":
    main()
