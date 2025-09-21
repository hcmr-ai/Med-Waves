#!/usr/bin/env python3
"""
Visualize Basin Feature

This script creates a visualization of the basin categorical feature
to show how it divides the geographic space.
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def create_basin_visualization():
    """Create a visualization of the basin feature."""
    
    # Create a grid of longitude and latitude points
    lon_range = np.linspace(-15, 40, 100)
    lat_range = np.linspace(30, 50, 80)
    lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)
    
    # Apply basin classification logic
    basin_grid = np.zeros_like(lon_grid, dtype=int)
    
    # Atlantic basin (longitude < -5)
    atlantic_mask = lon_grid < -5
    basin_grid[atlantic_mask] = 0
    
    # Eastern Mediterranean basin (longitude > 30)
    eastern_med_mask = lon_grid > 30
    basin_grid[eastern_med_mask] = 2
    
    # Mediterranean basin (everything else)
    mediterranean_mask = ~(atlantic_mask | eastern_med_mask)
    basin_grid[mediterranean_mask] = 1
    
    # Create the plot
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
    
    # Plot basin regions
    colors = ['red', 'green', 'blue']
    labels = ['Atlantic (0)', 'Mediterranean (1)', 'Eastern Med (2)']
    
    for basin_id in [0, 1, 2]:
        mask = basin_grid == basin_id
        if np.any(mask):
            ax.scatter(lon_grid[mask], lat_grid[mask], 
                      c=colors[basin_id], s=1, alpha=0.6, 
                      label=labels[basin_id], transform=ccrs.PlateCarree())
    
    # Add gridlines
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    
    # Set extent
    ax.set_extent([-15, 40, 30, 50], crs=ccrs.PlateCarree())
    
    # Add title and legend
    plt.title('Basin Categorical Feature\n(0=Atlantic, 1=Mediterranean, 2=Eastern Med)', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
    
    # Add text annotations
    ax.text(-8, 45, 'Atlantic\nBasin', transform=ccrs.PlateCarree(), 
            fontsize=12, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax.text(15, 40, 'Mediterranean\nBasin', transform=ccrs.PlateCarree(), 
            fontsize=12, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax.text(35, 35, 'Eastern Med\nBasin', transform=ccrs.PlateCarree(), 
            fontsize=12, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('basin_feature_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Basin Feature Visualization:")
    print("=" * 50)
    print("Basin ID | Basin Name        | Longitude Range")
    print("-" * 50)
    print("0        | Atlantic          | < -5°")
    print("1        | Mediterranean     | -5° to 30°")
    print("2        | Eastern Med       | > 30°")
    print("=" * 50)
    print("\nThis categorical feature provides explicit geographic context")
    print("to help the model understand different oceanographic basins.")

if __name__ == "__main__":
    create_basin_visualization()
