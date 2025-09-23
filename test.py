import pandas as pd

# lire uniquement le schéma, pas tout le fichier
df = pd.read_parquet("bronze\QUOT_departement_01_periode_1950-2023_autres-parametres.parquet")

print("Colonnes du parquet :")
print(df.columns.tolist())

# si tu veux voir aussi les 5 premières lignes
print(df.head())
