import pandas as pd

df = pd.read_parquet("data/silver/time_series/meteo_clean.parquet")

print(df.shape)
print(df['date'].min(), df['date'].max())
print("Stations :", df['NUM_POSTE'].nunique())
print(df[['date','NUM_POSTE','NOM_USUEL','RR','RR_7d','API','TM','TM_7d']].head(10))

# Taux de manquants par colonne (top 15)
print(df.isna().mean().sort_values(ascending=False).head(15))

# Combien de lignes sans info météo clé
key_cols = ['RR','TM','TN','TX']
print("Lignes sans aucune valeur parmi RR/TM/TN/TX :", df[key_cols].isna().all(axis=1).sum())

# Couverture par station (nombre de jours disponibles)
print(
    df.groupby('NUM_POSTE')['date'].count()
      .sort_values(ascending=False)
      .head(10)
)

# Part de lignes avec RR manquant vs disponible
tot = len(df)
print("RR non nuls :", df['RR'].notna().sum(), "/", tot)
print("TM non nuls :", df['TM'].notna().sum(), "/", tot)
