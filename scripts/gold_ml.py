#!/usr/bin/env python3
# gold_ml.py - Couche Gold (ML uniquement)
# Hackathon EFREI - Ville Durable et Intelligente
# Équipe : Prédiction Inondations Île-de-France

import pandas as pd
import numpy as np
import json
import joblib
from datetime import datetime, timedelta
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

# ML
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import lightgbm as lgb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/gold_ml.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GoldMLPipeline:
    def __init__(self):
        self.silver_path = Path("data/silver/basins/dataset_bassin_ml_ready.parquet")
        self.gold_path = Path("data/gold")
        self.models_path = Path("models")
        for p in [self.gold_path, self.models_path]:
            p.mkdir(parents=True, exist_ok=True)

    def load_silver_data(self):
        df = pd.read_parquet(self.silver_path)
        df["date"] = pd.to_datetime(df["date"])
        logger.info(f" Données chargées : {df.shape}")
        return df

    def prepare_datasets(self, df):
        # Exclure colonnes non numériques
        exclude_cols = [
            "date", "bassin_id", "station_id",
            "nom_station", "cours_eau", "risque_inondation",
            "niveau_j1", "niveau_j2", "niveau_j3",
            "alerte_j1", "alerte_j2", "alerte_j3"
        ]
        features = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ["int64", "float64", "bool"]]

        targets_reg = ["niveau_j1", "niveau_j2", "niveau_j3"]
        targets_clf = ["alerte_j1", "alerte_j2", "alerte_j3"]

        df = df.dropna(subset=features + targets_reg + targets_clf)

        logger.info(f" Features retenues ({len(features)}) : {features[:10]}...")
        return df, features, targets_reg, targets_clf


    def split_data(self, df):
        df = df.sort_values("date")
        n = len(df)
        train = df.iloc[:int(0.7*n)]
        val = df.iloc[int(0.7*n):int(0.85*n)]
        test = df.iloc[int(0.85*n):]
        return train, val, test

    def train_regression(self, train, val, features):
        models, results = {}, {}
        for horizon in ["j1", "j2", "j3"]:
            target = f"niveau_{horizon}"
            X_train, y_train = train[features], train[target]
            X_val, y_val = val[features], val[target]

            lgb_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.1, max_depth=8)
            lgb_model.fit(X_train, y_train)
            y_pred = lgb_model.predict(X_val)

            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)

            models[horizon] = lgb_model
            results[horizon] = {"rmse": rmse, "r2": r2}
            joblib.dump(lgb_model, self.models_path / f"regression_{horizon}.joblib")
            logger.info(f" Régression {horizon}: RMSE={rmse:.2f}, R²={r2:.3f}")
        return models, results

    def train_classification(self, train, val, features):
        models, results = {}, {}
        for horizon in ["j1", "j2", "j3"]:
            target = f"alerte_{horizon}"
            X_train, y_train = train[features], train[target]
            X_val, y_val = val[features], val[target]

            if len(y_train.unique()) < 2:
                continue

            lgb_clf = lgb.LGBMClassifier(n_estimators=150, class_weight="balanced")
            lgb_clf.fit(X_train, y_train)
            y_pred = lgb_clf.predict(X_val)
            auc = roc_auc_score(y_val, lgb_clf.predict_proba(X_val)[:, 1])
            report = classification_report(y_val, y_pred, output_dict=True)

            models[horizon] = lgb_clf
            results[horizon] = {"auc": auc, "f1": report["weighted avg"]["f1-score"]}
            joblib.dump(lgb_clf, self.models_path / f"classification_{horizon}.joblib")
            logger.info(f" Classification {horizon}: AUC={auc:.3f}, F1={report['weighted avg']['f1-score']:.3f}")
        return models, results

    def generate_predictions(self, df, models_reg, models_clf, features):
        latest = df.sort_values("date").groupby(["bassin_id", "station_id"]).tail(1)
        preds = []
        for _, row in latest.iterrows():
            X = pd.DataFrame([row[features]])
            record = {"bassin_id": row["bassin_id"], "station_id": row["station_id"], "date": row["date"]}
            for h, model in models_reg.items():
                record[f"pred_niveau_{h}"] = float(model.predict(X))
            for h, model in models_clf.items():
                record[f"pred_alerte_{h}"] = int(model.predict(X))
            preds.append(record)
        preds_df = pd.DataFrame(preds)
        preds_df.to_parquet(self.gold_path / "predictions.parquet", index=False)
        logger.info(" Prédictions sauvegardées")
        return preds_df


def main():
    gold = GoldMLPipeline()
    df = gold.load_silver_data()
    df_ml, features, targets_reg, targets_clf = gold.prepare_datasets(df)
    train, val, test = gold.split_data(df_ml)
    reg_models, reg_results = gold.train_regression(train, val, features)
    clf_models, clf_results = gold.train_classification(train, val, features)
    preds = gold.generate_predictions(df_ml, reg_models, clf_models, features)

    # Sauvegarde résultats
    results = {"regression": reg_results, "classification": clf_results}
    with open(gold.gold_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n GOLD ML terminé. Résultats et modèles sauvegardés.")


if __name__ == "__main__":
    main()
