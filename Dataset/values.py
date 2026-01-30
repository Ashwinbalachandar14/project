import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio
import glob
import os

# --- CONFIGURATION ---
BASE_DIR = r"f:\project\Dataset"
csv_path = os.path.join(BASE_DIR, "Groundwater.csv")
district_folder = os.path.join(BASE_DIR, "Pollachi")  # Start with one district

# 1. Load the Groundwater Data
print("Loading Groundwater CSV...")
df = pd.read_csv(csv_path)

# 2. CRITICAL STEP: Filter for Pollachi Coordinates
# Pollachi is approx Lat: 10.5-11.0, Lon: 76.9-77.2
print(f"Total rows in CSV: {len(df)}")
df_pollachi = df[
    (df['latitude'] > 10.0) & (df['latitude'] < 11.5) &
    (df['longitude'] > 76.5) & (df['longitude'] < 77.5)
].copy()

print(f"Rows found in Pollachi region: {len(df_pollachi)}")

if df_pollachi.empty:
    print("\nERROR: Your Groundwater.csv does not have any data for Pollachi!")
    print("Please check your input file. You need coordinates starting with Longitude 77.x")
    exit()

# 3. Setup GeoDataFrame
geometry = [Point(xy) for xy in zip(df_pollachi['longitude'], df_pollachi['latitude'])]
gdf = gpd.GeoDataFrame(df_pollachi, geometry=geometry)
gdf.crs = "EPSG:4326"

# 4. Find the Images
tif_files = glob.glob(os.path.join(district_folder, "**", "*.tiff"), recursive=True)
if not tif_files:
    tif_files = glob.glob(os.path.join(district_folder, "**", "*.tif"), recursive=True)

if not tif_files:
    print("No images found! Check path.")
    exit()

# 5. Extract Real Values
def get_val(img_path, gdf):
    with rasterio.open(img_path) as src:
        # Reproject points to match image (UTM)
        if gdf.crs != src.crs:
            gdf_projected = gdf.to_crs(src.crs)
        else:
            gdf_projected = gdf
            
        coord_list = [(pt.x, pt.y) for pt in gdf_projected.geometry]
        return [val[0] for val in src.sample(coord_list)]

# Auto-detect bands
bands = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
for band in bands:
    # Find file matching the band name
    match = [f for f in tif_files if f"_{band}_" in os.path.basename(f)]
    if match:
        print(f"Extracting {band}...")
        gdf[band] = get_val(match[0], gdf)

# 6. Calculate Indices (avoiding division by zero)
print("Calculating Indices...")
gdf['NDVI'] = (gdf['B08'] - gdf['B04']) / (gdf['B08'] + gdf['B04'] + 1e-5)
gdf['IOI'] = gdf['B04'] / (gdf['B02'] + 1e-5)
gdf['CMI'] = gdf['B11'] / (gdf['B08'] + 1e-5)

# 7. Save Corrected File
output_file = os.path.join(BASE_DIR, "Pollachi_Corrected_Values.csv")
gdf.to_csv(output_file, index=False)
print(f"\nSuccess! Real values saved to: {output_file}")
print(gdf[['latitude', 'longitude', 'NDVI', 'B02']].head())

