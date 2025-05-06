import io
import xarray as xr
from collections import defaultdict
import boto3
from datetime import datetime as dt
from tqdm import tqdm

def compute_daily_mean_from_keys_NETCDF(bucket_name, region_name, file_keys, variable=None):
    """
    Compute the daily mean of a list of .nc4 S3 file keys grouped by day.

    Parameters
    ----------
    bucket_name : str
        Name of the S3 bucket.
    region_name : str
        AWS region name where the bucket is hosted.
    file_keys : list of str
        List of S3 keys (paths) to .nc4 files.
    variable : str, optional
        Name of the variable to select from each dataset.
        If None, all variables are kept.

    Returns
    -------
    dict
        Dictionary where keys are datetime.date objects, and values are xarray.Dataset
        (or xarray.DataArray if single variable selected) representing the daily mean for each day.

    Notes
    -----
    This function:
    - Groups all files by date (YYYY-MM-DD)
    - Downloads files for each day
    - Loads them with xarray
    - Optionally selects a variable
    - Computes the daily mean over all 3-hourly files
    """
    s3_client = boto3.client('s3', region_name=region_name)

    # Step 1: Group file keys by day
    daily_files = defaultdict(list)

    for key in file_keys:
        filename = key.split('/')[-1]
        date_part = filename.split('.')[1][1:9]  # 'A20220501' -> '20220501'
        day = dt.strptime(date_part, "%Y%m%d").date()
        daily_files[day].append(key)

    # Step 2: Loop over each day, load all files, average
    daily_means = {}

    for day in tqdm(daily_files, desc="Processing days", unit="day"):
        keys = daily_files[day]
        datasets = []
        for key in keys:
            obj = s3_client.get_object(Bucket=bucket_name, Key=key)
            data = io.BytesIO(obj['Body'].read())
            ds = xr.open_dataset(data)

            if variable is not None:
                if variable in ds:
                    ds = ds[variable]
                else:
                    raise ValueError(f"Variable '{variable}' not found in file {key}.")

            datasets.append(ds)

        # Merge all hourly datasets for this day
        combined = xr.concat(datasets, dim='time')

        # Take mean over time dimension
        daily_mean = combined.mean(dim='time')

        daily_means[day] = daily_mean

    return daily_means
