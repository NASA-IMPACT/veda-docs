import re
from collections import defaultdict
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
import gc

class S3TimeAggregator:
    """
    Streams remote S3‐hosted NetCDF files, groups them by a regex on their URL,
    applies a reduction (e.g. daily/weekly/monthly mean), and stitches the results together.
    """

    def __init__(
        self,
        fs,
        urls,
        variable=None,
        engine="h5netcdf",
        chunks=None,
        group_regex=r"\.A(\d{8})\.",
        date_format="%Y%m%d",
        agg_frequency="D",
        agg_func=None,
        lat_range=None,
        lon_range=None,
    ):
        """
        Parameters
        ----------
        fs : fsspec‐style filesystem
            e.g., an S3FileSystem with proper credentials.
        variable : str or None
            Name of the variable to extract. If None, converts entire dataset → array.
        engine : str
            xarray engine, e.g. "h5netcdf" or "netcdf4".
        chunks : dict or None
            Dask chunking spec, e.g. {"time":1, "lat":500, "lon":500}.
        group_regex : str
            Regex with one capture group that extracts a date string from each URL.
        date_format : str
            `pd.to_datetime(..., format=date_format)` on the captured string.
        agg_frequency : str
            Resampling frequency: "D" for daily, "W" for weekly, "M" for monthly.
        agg_func : callable or None
            Function taking a DataArray ⇒ reduced DataArray.
            If None, defaults to `.mean(dim="time")`.
        lat_range : tuple(float, float) or None
            (lat_min, lat_max) to subset each file.
        lon_range : tuple(float, float) or None
            (lon_min, lon_max) to subset each file.
        """
        self.fs            = fs
        self.urls          = list(urls)
        self.variable      = variable
        self.engine        = engine
        self.chunks        = chunks or {"time": 1}
        self.group_pat     = re.compile(group_regex)
        self.date_format   = date_format
        self.agg_frequency = agg_frequency
        self.agg_func      = agg_func or (lambda da: da.mean(dim="time", keep_attrs=True))
        self.lat_range     = lat_range
        self.lon_range     = lon_range


    def group_urls(self, urls):
        """Group URLs by the specified frequency period (D, W, M, etc.)."""
        groups = defaultdict(list)
        for url in self.urls:
            m = self.group_pat.search(url)
            if not m:
                raise ValueError(f"URL did not match grouping pattern: {url}")
            date = pd.to_datetime(m.group(1), format=self.date_format)
            period = date.to_period(self.agg_frequency).to_timestamp()
            groups[period].append(url)
        return dict(groups)

    def _process_group(self, period, url_list):
        file_objs = []
        ds_list   = []
        pieces    = []

        for url in url_list:
            fobj = self.fs.open(url, "rb")
            file_objs.append(fobj)

            ds = xr.open_dataset(fobj, engine=self.engine, chunks=self.chunks)

            # ── spatial subset as early as possible ───────────────────
            if self.lat_range is not None:
                ds = ds.sel(lat=slice(self.lat_range[0], self.lat_range[1]))
            if self.lon_range is not None:
                ds = ds.sel(lon=slice(self.lon_range[0], self.lon_range[1]))

            ds_list.append(ds)

            da = ds[self.variable] if self.variable else ds.to_array()
            pieces.append(da)

        # concat & compute reduction
        stacked = xr.concat(pieces, dim="time")
        print(f"⏳ Processing period {period.date()} …")
        with ProgressBar():
            result = self.agg_func(stacked).compute()

        # clean up
        for ds in ds_list:
            ds.close()
        for f in file_objs:
            f.close()
        del file_objs, ds_list, pieces, stacked
        gc.collect()

        # stamp on time
        return result.expand_dims(time=[period])

    def run(self, urls=None):
        """
        Orchestrate grouping, per-period processing, and final concatenation.

        Parameters
        ----------
        urls : optional list of str
            If provided, overrides the urls passed at init.

        Returns
        -------
        xarray.DataArray or Dataset
        """
        to_group = urls if urls is not None else self.urls
        grouped  = self.group_urls(to_group)

        results = []
        for period, url_list in sorted(grouped.items()):
            results.append(self._process_group(period, url_list))

        # combine all period outputs along the new time dimension
        return xr.concat(results, dim="time")
