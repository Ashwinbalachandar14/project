import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio
import numpy as np
import glob
import os

# ================================
# 1. PATHS
# ================================
BASE_DIR = r"F:\project\Dataset"
csv_path = os.path.join(BASE_DIR, "Groundwater.csv")

print("\nLoading CGWB groundwater dataset...")

# ================================
# 2. LOAD GROUNDWATER CSV
# ================================
df = pd.read_csv(csv_path)

# Keep only required columns
df = df[['latitude', 'longitude', 'date', 'currentlevel']]

# Convert to numeric safely
df['latitude'] = pd.to_numeric(df['latitude'], errors="coerce")
df['longitude'] = pd.to_numeric(df['longitude'], errors="coerce")
df['currentlevel'] = pd.to_numeric(df['currentlevel'], errors="coerce")

# Drop invalid rows
df = df.dropna()

print("Total groundwater rows:", len(df))


# ================================
# 3. FILTER BY POLLACHI REGION (Bounding Box)
# ================================
# Pollachi + Coimbatore region approx coordinates
df = df[
    (df['latitude'] >= 10.5) & (df['latitude'] <= 11.3) &
    (df['longitude'] >= 76.6) & (df['longitude'] <= 77.4)
]

print("Rows inside Pollachi-Coimbatore region:", len(df))

if len(df) == 0:
    print("ERROR: No groundwater points found in this region.")
    exit()


# ================================
# 4. CONVERT TO GEODATAFRAME
# ================================
gdf = gpd.GeoDataFrame(
    df,
    geometry=[Point(xy) for xy in zip(df['longitude'], df['latitude'])],
    crs="EPSG:4326"
)


# ================================
# 5. FUNCTION TO SAMPLE RASTER
# ================================
def sample_raster(raster_path, points_gdf):
    with rasterio.open(raster_path) as src:

        # Convert points CRS to raster CRS
        if points_gdf.crs != src.crs:
            points_gdf = points_gdf.to_crs(src.crs)

        coords = [(pt.x, pt.y) for pt in points_gdf.geometry]

        sampled_vals = []
        for val in src.sample(coords):
            sampled_vals.append(val[0])

        return sampled_vals


# ================================
# 6. SENTINEL BAND EXTRACTION
# ================================
print("\nScanning Sentinel images...")

district_folder = os.path.join(BASE_DIR, "Pollachi")

# Find all tif files
tif_files = glob.glob(os.path.join(district_folder, "**", "*.tif*"), recursive=True)

if not tif_files:
    print("ERROR: No Sentinel TIFF images found!")
    exit()

# Required bands
target_bands = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']

# Extract each band
for band in target_bands:
    match = [f for f in tif_files if f"_{band}_" in os.path.basename(f)]

    if match:
        print("Extracting", band, "from:", os.path.basename(match[0]))
        gdf[band] = sample_raster(match[0], gdf)
    else:
        print("WARNING: Band not found:", band)


# ================================
# 7. FIX SENTINEL SCALE VALUES
# Sentinel pixels are scaled by 10000
# ================================
for band in target_bands:
    if band in gdf.columns:
        gdf[band] = gdf[band] / 10000.0


# ================================
# 8. CALCULATE INDICES
# ================================

# NDVI
gdf["NDVI"] = (gdf["B08"] - gdf["B04"]) / (gdf["B08"] + gdf["B04"] + 1e-10)

# IOI
gdf["IOI"] = gdf["B04"] / (gdf["B02"] + 1e-10)

# CMI
gdf["CMI"] = gdf["B11"] / (gdf["B08"] + 1e-10)


# ================================
# 9. TARGET VARIABLE IN FEET
# ================================
gdf["currentlevel_ft"] = gdf["currentlevel"] * 3.281


# ================================
# 10. SAVE FINAL OUTPUT
# ================================
output_file = os.path.join(BASE_DIR, "final_groundwater_features.csv")

gdf.drop(columns="geometry").to_csv(output_file, index=False)

print("\nSUCCESS! Dataset saved at:")
print(output_file)

print("\nSample Output:")
print(gdf.head())
