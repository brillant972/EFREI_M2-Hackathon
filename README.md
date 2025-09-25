# README — Prédiction d’épisodes pluvieux en Île-de-France (météo only)

## Objectif

Construire un pipeline de données simple et reproductible pour :

1. ingérer et normaliser des observations météo quotidiennes d’Île-de-France,
2. nettoyer et enrichir ces données (roulantes, indices, saisonnalité),
3. préparer un dataset exploitable pour l’exploration et, ensuite, pour un modèle ML météo-only (ex. prédire RR demain ou détecter un jour “pluvieux fort”).

---

## Données sources

* Plateforme : data.gouv.fr
* Dataset météo quotidien : `6569b51ae64326786e4e8e1a`
* Périmètre retenu : départements d’Île-de-France (75, 77, 78, 91, 92, 93, 94, 95)
* Périodes : `1950-2023` et `2024-2025`
* Blocs : `RR-T-Vent` et `autres-parametres`
* Format d’origine : CSV.gz quotidiens par département.
* Écriture : conversion en **Parquet dataset** (dossier `part_*.parquet`) pour lecture rapide et incrémentale.

---

## Architecture

### Couche Bronze — Ingestion

**Ce que fait le script `bronze.py` :**

* Interroge l’API data.gouv pour lister les ressources du dataset cible.
* Filtre strictement les fichiers par motif de nom (parse des schémas `Q_...`, `QUOT_...`, etc.), période, bloc et code département en IDF.
* Télécharge uniquement ce qui matche et **convertit en Parquet par chunks** pour limiter la mémoire :

  * Entrée : `.../Q_{dept}_{période}_{bloc}.csv.gz`
  * Sortie : `data/bronze/meteo/Q_{dept}_{période}_{bloc}_parquet/part_*.parquet`
* Idempotent : si le dossier Parquet existe déjà, il est laissé tel quel.

**Sorties Bronze (exemples) :**

```
data/bronze/meteo/
  Q_75_1950-2023_RR-T-Vent_parquet/part_0.parquet
  Q_75_1950-2023_autres-parametres_parquet/part_1.parquet
  ...
```

### Couche Silver — Nettoyage & Feature engineering (météo uniquement)

**Ce que fait le script `silver.py` :**

* **Lecture récursive** de tous les Parquet (ou CSV.gz si pas de Parquet) sous `data/bronze/meteo/`.
* **Colonnes cibles gardées** si présentes :
  `NUM_POSTE, NOM_USUEL, LAT, LON, ALTI, date/AAAAMMJJ, TN, TX, TM, RR, PMER, FFM`.
* **Normalisation types** :

  * `date` dérivée de `AAAAMMJJ` si besoin.
  * `NUM_POSTE` en chaîne (préserve les zéros).
  * Numériques convertis et **downcast** (`float32`/`Int32`) pour réduire la RAM.
  * `NOM_USUEL` en catégorie.
* **Nettoyage** :

  * Déduplication par (`NUM_POSTE`,`date`) si possible.
  * **Bornes plausibles** appliquées (ex. RR < 0 ou > 300 → NaN ; TX > 55 → NaN, etc.).
  * Tri par station/date puis **interpolation courte** par station (`limit=3`) sur `TN, TX, TM, RR, PMER, FFM`.
  * Suppression de colonnes **quasi vides** (>95% NaN) ou **constantes** (hors colonnes clés).
  * Suppression de lignes **sans aucune** info météo utile parmi `[RR, TM, TN, TX]`.
  * Écrit un **rapport de nettoyage** : `data/silver/features/cleaning_report.json`.
* **Features ajoutées** :

  * Pluie : `RR_7d`, `RR_14d`, `RR_30d` (cumuls), `API` (Antecedent Precipitation Index, k=0.9).
  * Température : `TM_7d`, `TM_30d` (moyennes glissantes).
  * Calendrier : `year`, `month`, `day_of_year`, `sin_doy`, `cos_doy`.

**Sorties Silver :**

```
data/silver/time_series/meteo_clean.parquet
data/silver/features/features_summary.json
data/silver/features/cleaning_report.json
```

---

## Dictionnaire des colonnes (Silver)

| Colonne                    | Description                                | Unité / Remarque   |
| -------------------------- | ------------------------------------------ | ------------------ |
| NUM\_POSTE                 | Code station météo                         | chaîne             |
| NOM\_USUEL                 | Nom usuel de la station                    | catégorie          |
| LAT, LON                   | Latitude, longitude                        | degrés (float32)   |
| ALTI                       | Altitude                                   | mètres             |
| date                       | Date de l’observation                      | datetime           |
| TN, TM, TX                 | Températures mini / moyenne / maxi du jour | °C                 |
| RR                         | Pluie du jour                              | mm                 |
| PMER                       | Pression au niveau mer                     | hPa                |
| FFM                        | Vent (vitesse moyenne)                     | m/s (selon source) |
| RR\_7d, RR\_14d, RR\_30d   | Cumuls pluie 7/14/30 jours                 | mm                 |
| API                        | Indice pluie antécédente (k=0.9)           | sans unité         |
| TM\_7d, TM\_30d            | Moyennes glissantes de TM                  | °C                 |
| year, month, day\_of\_year | Calendrier                                 | entiers            |
| sin\_doy, cos\_doy         | Encodage cyclique du jour de l’année       | \[-1, 1]           |

---

## EDA — Liste des graphiques et lecture recommandée

1. **Nombre d’observations par année**
   Mesure la couverture temporelle globale et les ruptures éventuelles de séries.

2. **Top 20 stations par nombre d’observations**
   Identifie les stations les plus complètes pour des analyses ou validations ciblées.

3. **Taux de valeurs manquantes (Top 15 colonnes)**
   Met en évidence les variables les plus incomplètes pour adapter l’usage en ML.

4. **Distribution de RR (mm) — clip à 100 mm**
   Explore la répartition des pluies journalières et la queue des fortes valeurs.

5. **Distribution de RR\_7d (mm) — clip à 200 mm**
   Observe la distribution des cumuls glissants sur 7 jours (épisodes prolongés).

6. **RR\_7d et API — station la plus fournie**
   Visualise la dynamique des cumuls et de l’indice d’antécédence sur une station de référence.

7. **RR moyen par mois (toutes années)**
   Donne la saisonnalité moyenne de la pluie à l’échelle mensuelle.

8. **Part des jours de pluie (RR ≥ 1 mm) par année**
   Suit l’évolution interannuelle de la fréquence des jours pluvieux.

9. **Part des jours de pluie forte (RR ≥ 10 mm) par année**
   Suit la fréquence interannuelle des épisodes plus intenses.

10. **RR moyen par jour de l’année (climatologie)**
    Trace le cycle saisonnier fin, jour par jour, toutes années confondues.

11. **Corrélation Spearman des features avec RR\_j1**
    Approxime la pertinence de chaque feature pour prédire la pluie du lendemain (RR+1).

12. **Matrice de corrélations Spearman (features météo/calendaires)**
    Met en évidence les redondances et dépendances entre variables pour guider la sélection de features.

> Les graphiques sont pensés pour de grands volumes. Selon la machine, on peut échantillonner (déjà prévu dans le code EDA).

---

## Étapes ML envisagées (météo only, prochaines itérations)

* **Cibles** possibles :

  * Régression : `RR_j1` (pluie J+1) construite par `groupby(NUM_POSTE)` + `shift(-1)`.
  * Classification : jour pluvieux fort (ex. `RR_j1 ≥ seuil_station`, p95).
* **Features candidates** : `RR` présents et cumulés (`RR_7d/14d/30d`), `API`, `TM` et ses moyennes glissantes, encodage saisonnier (`sin_doy`, `cos_doy`), éventuellement `PMER`, `FFM`.
* **Découpage temporel** : split train/val/test **par date** (pas aléatoire).
* **Métriques** : MAE/RMSE pour régression, AUC/PR pour classification déséquilibrée.

---

## Exécution

1. Bronze

```bash
python scripts/bronze.py
# Sorties dans data/bronze/meteo/*_parquet/part_*.parquet
```

2. Silver

```bash
python scripts/silver.py
# Sorties:
# - data/silver/time_series/meteo_clean.parquet
# - data/silver/features/cleaning_report.json
# - data/silver/features/features_summary.json
```

3. EDA (notebook)
   Ouvrir le notebook EDA et exécuter les cellules. Les chemins pointent vers `data/silver/time_series/meteo_clean.parquet`.

---

## Points d’attention

* Les sources historiques sont hétérogènes : certaines stations ont des séries longues avec des trous. Le rapport de nettoyage détaille les colonnes supprimées et les lignes sans info utile.
* Les unités des colonnes “vent” peuvent varier selon la source ; pour un usage ML, privilégier d’abord RR, cumulés, API et température.
* La couche Silver est **idempotente** : relancer écrase `meteo_clean.parquet` et met à jour les rapports.