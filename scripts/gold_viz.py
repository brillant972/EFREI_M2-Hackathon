#!/usr/bin/env python3
# gold_viz.py - Couche Gold (Visualisation & Rapport)
# Hackathon EFREI - Ville Durable et Intelligente

import pandas as pd
import json
import matplotlib.pyplot as plt
import folium
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/gold_viz.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GoldVisualization:
    def __init__(self):
        self.gold_path = Path("data/gold")
        self.viz_path = Path("visualizations")
        self.reports_path = Path("reports")
        for p in [self.viz_path, self.reports_path]:
            p.mkdir(parents=True, exist_ok=True)

    def load_results(self):
        with open(self.gold_path / "results.json") as f:
            results = json.load(f)
        preds = pd.read_parquet(self.gold_path / "predictions.parquet")
        return results, preds

    def generate_charts(self, results):
        if "regression" in results:
            rmse = [v["rmse"] for v in results["regression"].values()]
            plt.bar(["j1", "j2", "j3"], rmse)
            plt.title("RMSE régression")
            plt.savefig(self.viz_path / "rmse_regression.png")
            plt.close()
        logger.info("Graphiques sauvegardés")

    def generate_map(self, preds):
        m = folium.Map(location=[48.8566, 2.3522], zoom_start=10)
        for _, row in preds.iterrows():
            folium.CircleMarker(
                location=[48.85, 2.35],
                radius=10,
                color="red" if row["pred_alerte_j1"] else "green",
                popup=f"{row['bassin_id']} - Niveau prévu J+1: {row['pred_niveau_j1']:.1f} cm"
            ).add_to(m)
        map_file = self.viz_path / "carte_risque.html"
        m.save(str(map_file))
        logger.info(f"Carte interactive sauvegardée -> {map_file}")

    def generate_report(self, results, preds):
        report_file = self.reports_path / "rapport_final.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(f"# Rapport Hackathon\n\n")
            f.write(f"Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n")
            f.write("## Résultats ML\n")
            f.write(json.dumps(results, indent=2))
            f.write("\n\n## Prédictions\n")
            f.write(str(preds.head()))
        logger.info(f"Rapport généré -> {report_file}")


def main():
    viz = GoldVisualization()
    results, preds = viz.load_results()
    viz.generate_charts(results)
    viz.generate_map(preds)
    viz.generate_report(results, preds)
    print("\n GOLD Viz terminé. Graphiques, carte et rapport sauvegardés.")


if __name__ == "__main__":
    main()
