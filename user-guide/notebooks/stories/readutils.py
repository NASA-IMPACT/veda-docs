import io
import xarray as xr
from collections import defaultdict
import boto3
from datetime import datetime as dt
from typing import Dict, List, Optional
from tqdm import tqdm
import re
from pystac_client import Client


def list_files(bucket_name, prefix, region, file_extension):
    """
    List all files from the S3 bucket for a given year.

    Parameters
    ----------
    bucket_name : str
        The S3 bucket name.
    prefix : str
        The object group within S3 bucket
    region : str, optional
        AWS region where the bucket is located.
    year : int or str
        The year to list files for (e.g., 2022).
    file extension: str
        Tells what type of file format is present (e.g., .nc4)

    Returns
    -------
    list of str
        List of S3 keys (file paths) corresponding to .nc4 files for the given year.

    """
    s3 = boto3.client('s3', region_name=region)

    all_keys = []
    continuation_token = None

    while True:
        if continuation_token:
            response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, ContinuationToken=continuation_token)
        else:
            response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        
        contents = response.get('Contents', [])
        for obj in contents:
            if obj['Key'].endswith(file_extension):
                all_keys.append(obj['Key'])

        if response.get('IsTruncated'):
            continuation_token = response['NextContinuationToken']
        else:
            break

    return all_keys


def list_files_MODIS(
    bucket_name: str,
    prefix: str,
    region: str,
    file_extension: str,
    year: Optional[int] = None
) -> List[str]:
    """
    List all S3 object keys under a given prefix (and optional year‐filter) ending with file_extension.

    Parameters
    ----------
    bucket_name : str
        The S3 bucket name (e.g. "lp-prod-protected").
    prefix : str
        The key prefix within the bucket (e.g. "MCD19A2.061").
    region : str
        AWS region where the bucket is located (e.g. "us-west-2").
    file_extension : str
        File extension to match (e.g. ".nc4" or ".hdf").
    year : int, optional
        If provided, only return keys whose filename contains the pattern
        `.A<YYYY><DDD>.` (where DDD is the day of year), ensuring the file
        belongs to that calendar year.

    Returns
    -------
    List[str]
        List of matching S3 keys.

    Raises
    ------
    botocore.exceptions.BotoCoreError / boto3.errors.S3Error
        If listing or pagination fails.
    """
    s3 = boto3.client('s3', region_name=region)

    # compile a year‐filter regex if requested
    pattern = None
    if year is not None:
        # matches “.A2022DDD.” where DDD is any three digits
        pattern = re.compile(rf"\.A{year}\d{{3}}\.")  

    all_keys: List[str] = []
    continuation_token = None

    while True:
        if continuation_token:
            resp = s3.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix,
                ContinuationToken=continuation_token
            )
        else:
            resp = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

        for obj in resp.get('Contents', []):
            key = obj['Key']
            if not key.endswith(file_extension):
                continue
            if pattern and not pattern.search(key):
                continue
            all_keys.append(key)

        if resp.get('IsTruncated'):  # more pages
            continuation_token = resp['NextContinuationToken']
        else:
            break

    return all_keys


def load_datasets_from_keys(
    bucket_name: str,
    region_name: str,
    file_keys: List[str],
    data_source: str,
    variable: Optional[str] = None,
    
) -> Dict[dt.date, List[xr.Dataset]]:
    """
    Load a batch of NetCDF4 files from S3 into xarray, grouping them by date.

    This function:
      - Parses each key’s filename to extract the YYYYMMDD date,
      - Downloads each object from the specified S3 bucket,
      - Opens it as an xarray.Dataset (or selects a single DataArray if `variable` is given),
      - Groups the resulting Dataset/DataArray objects into a dict keyed by date.

    Args:
        bucket_name: Name of the S3 bucket containing the .nc4 files.
        region_name: AWS region of the S3 bucket.
        file_keys: List of S3 object keys (e.g. `"path/to/A20220501.somefile.nc4"`).
        variable: If provided, selects only this variable from each dataset; 
                  otherwise the full dataset is returned.

    Returns:
        A dict mapping each `datetime.date` to a list of xarray.Dataset (or DataArray)
        objects corresponding to all files from that day.

    Raises:
        ValueError: If `variable` is specified but not found in a dataset.
        botocore.exceptions.BotoCoreError / boto3 errors: if the S3 download fails.
    """
    # Initialize S3 client
    s3_client = boto3.client('s3', region_name=region_name)

    # Group keys by their calendar date
    daily_files: Dict[dt.date, List[str]] = defaultdict(list)
    for key in file_keys:
        # Assumes filename contains 'AYYYYMMDD' immediately after first dot
        filename = key.split('/')[-1]
        if data_source == 'GLDAS':
            date_str = filename.split('.')[1][1:9]
        elif data_source == 'MERRA':
            date_str = filename.split('.')[-2]
        day = dt.strptime(date_str, "%Y%m%d").date()
        daily_files[day].append(key)

    # Download and open each file, grouping by day
    daily_datasets: Dict[dt.date, List[xr.Dataset]] = {}
    for day, keys in tqdm(daily_files.items(), desc="Loading daily datasets", unit="day"):
        datasets: List[xr.Dataset] = []
        for key in keys:
            obj = s3_client.get_object(Bucket=bucket_name, Key=key)
            buf = io.BytesIO(obj['Body'].read())
            ds = xr.open_dataset(buf)
            if variable is not None:
                if variable not in ds:
                    raise ValueError(f"Variable '{variable}' not found in file {key}")
                ds = ds[[variable]]  # keep as Dataset with one variable
            datasets.append(ds)
        daily_datasets[day] = datasets

    return daily_datasets


def list_stac_providers(stac_api_url):
    root = Client.open(stac_api_url)
    providers = [p.id for p in root.get_children()]
    providers = sorted(providers)



