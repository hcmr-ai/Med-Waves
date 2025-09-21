"""
Region mapping utility for consistent region handling across the codebase.

This module provides centralized region mapping between string names and integer IDs
for performance optimization while maintaining readability in plots and logs.
"""

from typing import Dict, Union, List
import logging

logger = logging.getLogger(__name__)

# Region mapping: string -> integer
REGION_TO_ID = {
    "atlantic": 0,
    "mediterranean": 1,
    "eastern_med": 2
}

# Reverse mapping: integer -> string
ID_TO_REGION = {v: k for k, v in REGION_TO_ID.items()}

# Region colors for plotting
REGION_COLORS = {
    0: '#FF6B6B',      # atlantic - red
    1: '#4ECDC4',      # mediterranean - teal
    2: '#45B7D1'       # eastern_med - blue
}

# Region display names for plots
REGION_DISPLAY_NAMES = {
    0: "Atlantic",
    1: "Mediterranean", 
    2: "Eastern Mediterranean"
}


class RegionMapper:
    """Utility class for region mapping operations."""
    
    @staticmethod
    def get_region_id(region_name: str) -> int:
        """
        Convert region name to integer ID.
        
        Args:
            region_name: Region name ("atlantic", "mediterranean", "eastern_med")
            
        Returns:
            Integer region ID (0, 1, 2)
            
        Raises:
            ValueError: If region name is not recognized
        """
        if region_name not in REGION_TO_ID:
            raise ValueError(f"Unknown region: {region_name}. Valid regions: {list(REGION_TO_ID.keys())}")
        return REGION_TO_ID[region_name]
    
    @staticmethod
    def get_region_name(region_id: int) -> str:
        """
        Convert region ID to region name.
        
        Args:
            region_id: Integer region ID (0, 1, 2)
            
        Returns:
            Region name ("atlantic", "mediterranean", "eastern_med")
            
        Raises:
            ValueError: If region ID is not recognized
        """
        if region_id not in ID_TO_REGION:
            raise ValueError(f"Unknown region ID: {region_id}. Valid IDs: {list(ID_TO_REGION.keys())}")
        return ID_TO_REGION[region_id]
    
    @staticmethod
    def get_display_name(region_id: int) -> str:
        """
        Convert region ID to display name for plots.
        
        Args:
            region_id: Integer region ID (0, 1, 2)
            
        Returns:
            Display name ("Atlantic", "Mediterranean", "Eastern Mediterranean")
        """
        return REGION_DISPLAY_NAMES.get(region_id, f"Region {region_id}")
    
    @staticmethod
    def get_region_color(region_id: int) -> str:
        """
        Get color for region ID in plots.
        
        Args:
            region_id: Integer region ID (0, 1, 2)
            
        Returns:
            Hex color code
        """
        return REGION_COLORS.get(region_id, '#808080')  # Default gray
    
    @staticmethod
    def convert_region_list_to_ids(region_names: List[str]) -> List[int]:
        """
        Convert list of region names to list of region IDs.
        
        Args:
            region_names: List of region names
            
        Returns:
            List of region IDs
        """
        return [RegionMapper.get_region_id(name) for name in region_names]
    
    @staticmethod
    def convert_region_list_to_names(region_ids: List[int]) -> List[str]:
        """
        Convert list of region IDs to list of region names.
        
        Args:
            region_ids: List of region IDs
            
        Returns:
            List of region names
        """
        return [RegionMapper.get_region_name(rid) for rid in region_ids]
    
    @staticmethod
    def get_all_region_ids() -> List[int]:
        """Get all valid region IDs."""
        return list(ID_TO_REGION.keys())
    
    @staticmethod
    def get_all_region_names() -> List[str]:
        """Get all valid region names."""
        return list(REGION_TO_ID.keys())
    
    @staticmethod
    def is_valid_region_id(region_id: Union[int, str]) -> bool:
        """
        Check if region ID or name is valid.
        
        Args:
            region_id: Region ID (int) or name (str)
            
        Returns:
            True if valid, False otherwise
        """
        if isinstance(region_id, int):
            return region_id in ID_TO_REGION
        elif isinstance(region_id, str):
            return region_id in REGION_TO_ID
        return False


def create_region_mapping_dict() -> Dict[str, int]:
    """
    Create a dictionary for mapping region names to IDs.
    Useful for Polars map_elements operations.
    
    Returns:
        Dictionary mapping region names to IDs
    """
    return REGION_TO_ID.copy()


def get_region_mapping_info() -> Dict[str, any]:
    """
    Get comprehensive region mapping information.
    
    Returns:
        Dictionary with all mapping information
    """
    return {
        "region_to_id": REGION_TO_ID,
        "id_to_region": ID_TO_REGION,
        "region_colors": REGION_COLORS,
        "display_names": REGION_DISPLAY_NAMES,
        "all_ids": list(ID_TO_REGION.keys()),
        "all_names": list(REGION_TO_ID.keys())
    }


# Convenience functions for backward compatibility
def region_name_to_id(region_name: str) -> int:
    """Convert region name to ID (convenience function)."""
    return RegionMapper.get_region_id(region_name)


def region_id_to_name(region_id: int) -> str:
    """Convert region ID to name (convenience function)."""
    return RegionMapper.get_region_name(region_id)


def region_id_to_display_name(region_id: int) -> str:
    """Convert region ID to display name (convenience function)."""
    return RegionMapper.get_display_name(region_id)
