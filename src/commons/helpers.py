from typing import List, Union

import numpy as np


class SeasonHelper:
    """Helper class for mapping timestamps/months to seasons.

    Seasons are defined as:
        - Winter: December, January, February (12, 1, 2)
        - Spring: March, April, May (3, 4, 5)
        - Summer: June, July, August (6, 7, 8)
        - Autumn: September, October, November (9, 10, 11)
    """

    # Class constant for season-month mapping
    SEASON_MONTHS = {
        "winter": [12, 1, 2],
        "spring": [3, 4, 5],
        "summer": [6, 7, 8],
        "autumn": [9, 10, 11]
    }

    @staticmethod
    def get_season_from_month(month: int) -> str:
        """Get season from month number (1-12).

        Args:
            month: Month number (1-12)

        Returns:
            Season name as string ('winter', 'spring', 'summer', 'autumn')
        """
        for season, months in SeasonHelper.SEASON_MONTHS.items():
            if month in months:
                return season
        return "unknown"

    @staticmethod
    def get_seasons_from_timestamps(timestamps: np.ndarray) -> List[str]:
        """Extract seasons from numpy datetime64 timestamps.

        Args:
            timestamps: Numpy array of datetime64 timestamps

        Returns:
            List of season names corresponding to each timestamp
        """
        months = timestamps.astype('datetime64[M]').astype(int) % 12 + 1
        return [SeasonHelper.get_season_from_month(month) for month in months]

    @staticmethod
    def get_seasons_from_months(months: Union[List[int], np.ndarray]) -> List[str]:
        """Get seasons from month numbers.

        Args:
            months: List or array of month numbers (1-12)

        Returns:
            List of season names corresponding to each month
        """
        if isinstance(months, np.ndarray):
            months = months.tolist()
        return [SeasonHelper.get_season_from_month(month) for month in months]

    @staticmethod
    def count_seasons(timestamps: np.ndarray) -> dict:
        """Count occurrences of each season in timestamps.

        Args:
            timestamps: Numpy array of datetime64 timestamps

        Returns:
            Dictionary with season counts: {'winter': count, 'spring': count, ...}
        """
        seasons = SeasonHelper.get_seasons_from_timestamps(timestamps)
        counts = {season: 0 for season in SeasonHelper.SEASON_MONTHS.keys()}
        for season in seasons:
            if season in counts:
                counts[season] += 1
        return counts
