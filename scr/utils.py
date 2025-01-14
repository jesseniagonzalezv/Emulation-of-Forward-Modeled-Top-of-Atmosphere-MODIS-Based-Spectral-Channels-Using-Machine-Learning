
import numpy as np
import pandas as pd
import xarray as xr

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import defaultdict


def format_times(times):
    # Function to format times
    # Group times by day
    times_by_day = defaultdict(list)
    for time in pd.to_datetime(times):
        day = f"{time.month:02d}-{time.day:02d}"
        times_by_day[day].append(f"T{time.hour:02d}")

    # Combine times for each day
    formatted_times = [f"{day}/" + ", ".join(times) for day, times in times_by_day.items()]
    return formatted_times


