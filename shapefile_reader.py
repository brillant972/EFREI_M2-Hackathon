import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the flood zones
flood_zones = gpd.read_file(r'C:\Users\ozann\OneDrive\Documents\Efrei_Travaux\M2\Hackaton\EFREI_M2-Hackathon\data\Zone_inondable_-_SDRIF-E\Zone_inondable_-_SDRIF-E.shp')

# Basic exploration
print(f"Shape: {flood_zones.shape}")
print(f"Columns: {flood_zones.columns.tolist()}")
print(f"CRS: {flood_zones.crs}")
print(f"Bounds: {flood_zones.bounds}")

print("\nFirst 3 rows:")
print(flood_zones.head(3))

print(f"\nTotal area covered: {flood_zones.area.sum():.2f} square units")

# Create a simple plot
fig, ax = plt.subplots(figsize=(12, 8))
flood_zones.plot(ax=ax, color='lightblue', edgecolor='blue', alpha=0.7)
ax.set_title("Flood Zones - Île-de-France (SDRIF-E)", fontsize=16)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.tight_layout()
plt.show()