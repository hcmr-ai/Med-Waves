import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import torch
from pathlib import Path


def visualize_region_classification(file_path, output_dir="./region_maps"):
    """
    Load a file and visualize which pixels are Atlantic vs Mediterranean.
    """
    # Load file
    if file_path.startswith("s3://"):
        import s3fs
        fs = s3fs.S3FileSystem()
        with fs.open(file_path, "rb") as f:
            data = torch.load(f, map_location="cpu")
    else:
        data = torch.load(file_path, map_location="cpu", weights_only=False)
    
    tensor = data["tensor"]
    feature_cols = data["feature_cols"]
    
    # Extract coordinates
    lat_idx = feature_cols.index("latitude")
    lon_idx = feature_cols.index("longitude")
    
    if tensor.ndim == 3:  # Hourly: (H, W, C)
        lat_data = tensor[..., lat_idx].numpy()
        lon_data = tensor[..., lon_idx].numpy()
    else:  # Daily: (T, H, W, C) - use first timestep
        lat_data = tensor[0, ..., lat_idx].numpy()
        lon_data = tensor[0, ..., lon_idx].numpy()
    
    # Classify each pixel
    # 0 = Atlantic (lon < -5.5), 1 = Mediterranean (lon >= -5.5)
    GIBRALTAR_LON = -5.5
    region_map = np.full_like(lon_data, np.nan)
    
    # Atlantic = 0, Mediterranean = 1
    region_map[lon_data < GIBRALTAR_LON] = 0  # Atlantic
    region_map[lon_data >= GIBRALTAR_LON] = 1  # Mediterranean
    
    # Handle NaN values
    region_map[np.isnan(lat_data) | np.isnan(lon_data)] = np.nan
    
    # Count pixels per region
    atlantic_pixels = np.sum(region_map == 0)
    med_pixels = np.sum(region_map == 1)
    total_valid = atlantic_pixels + med_pixels
    
    print(f"\n=== File: {file_path} ===")
    print(f"Atlantic pixels: {atlantic_pixels} ({atlantic_pixels/total_valid*100:.1f}%)")
    print(f"Mediterranean pixels: {med_pixels} ({med_pixels/total_valid*100:.1f}%)")
    
    # Create visualization
    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Plot region classification
    im = ax.pcolormesh(lon_data, lat_data, region_map, 
                       transform=ccrs.PlateCarree(),
                       cmap='RdBu_r',  # Red=Atlantic, Blue=Med
                       vmin=0, vmax=1,
                       shading='auto')
    
    # Add coastlines and features
    ax.coastlines(resolution='10m', linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    
    # Mark Gibraltar Strait
    GIBRALTAR_LAT = 36.13  # ~36°08'N
    ax.plot([GIBRALTAR_LON, GIBRALTAR_LON], [30, 45], 
            'g--', linewidth=2, transform=ccrs.PlateCarree(), 
            label=f'Gibraltar boundary (lon={GIBRALTAR_LON}°)')
    ax.plot(GIBRALTAR_LON, GIBRALTAR_LAT, 'g*', 
            markersize=15, transform=ccrs.PlateCarree(),
            label='Gibraltar Strait')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                       pad=0.05, shrink=0.7)
    cbar.set_label('Region: 0=Atlantic, 1=Mediterranean', fontsize=12)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Atlantic', 'Mediterranean'])
    
    ax.legend(loc='upper right')
    ax.set_title(f'Region Classification\n{file_path.split("/")[-1]}')
    ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
    
    # Save figure
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filename = Path(file_path).stem
    output_path = Path(output_dir) / f"{filename}_region_map.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved map to: {output_path}")
    plt.close()
    
    return region_map, atlantic_pixels, med_pixels

def main():
    # Configuration - specify a single file to visualize
    file_path = "/opt/dlami/nvme/preprocessed_subsampled_step_5/WAVEAN20231231.pt"
    
    print(f"\nVisualizing file: {file_path}")
    visualize_region_classification(file_path)


if __name__ == "__main__":
    main()