import rasterio
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1️⃣ Load all bands
# -------------------------------
paths = [
    r'F:\project\Dataset\kinathukaduv\2026-01-22-00_00_2026-01-22-23_59_Sentinel-2_L2A_B02_(Raw).tiff',
    r'F:\project\Dataset\kinathukaduv\2026-01-22-00_00_2026-01-22-23_59_Sentinel-2_L2A_B03_(Raw).tiff',
    r'F:\project\Dataset\kinathukaduv\2026-01-22-00_00_2026-01-22-23_59_Sentinel-2_L2A_B04_(Raw).tiff',
    r'F:\project\Dataset\kinathukaduv\2026-01-22-00_00_2026-01-22-23_59_Sentinel-2_L2A_B08_(Raw).tiff',
    r'F:\project\Dataset\kinathukaduv\2026-01-22-00_00_2026-01-22-23_59_Sentinel-2_L2A_B11_(Raw).tiff',
    r'F:\project\Dataset\kinathukaduv\2026-01-22-00_00_2026-01-22-23_59_Sentinel-2_L2A_B12_(Raw).tiff'
]

bands = []
for path in paths:
    with rasterio.open(path) as src:
        bands.append(src.read(1))  # read single band as 2D array

# -------------------------------
# 2️⃣ Stack for CNN
# -------------------------------
input_patch = np.stack(bands, axis=0)
print("CNN Input Shape:", input_patch.shape)  # (6, H, W)

# -------------------------------
# 3️⃣ Show True Color Image (RGB)
# -------------------------------
rgb = np.stack([bands[2], bands[1], bands[0]], axis=-1)  # B04, B03, B02
rgb = rgb / np.max(rgb)  # normalize to 0-1 for display
plt.figure(figsize=(8, 8))
plt.imshow(rgb)
plt.title('True Color Image (RGB)')
plt.axis('off')
plt.show()

# -------------------------------
# 4️⃣ Show all 6 bands individually
# -------------------------------
titles = ['B02 - Blue', 'B03 - Green', 'B04 - Red', 'B08 - NIR', 'B11 - SWIR1', 'B12 - SWIR2']
plt.figure(figsize=(15, 8))
for i, (b, t) in enumerate(zip(bands, titles), 1):
    plt.subplot(2, 3, i)
    plt.imshow(b, cmap='gray')
    plt.title(t)
    plt.axis('off')
plt.tight_layout()
plt.show()
