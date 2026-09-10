#!/usr/bin/env python3
"""Download ERA5-Land land-surface state for 2004-01-01 00:00 UTC.

Pulls the variables needed to construct an HTESSEL restart initial state
over the CMFD China domain (73.35-135.05 E, 18.15-53.55 N, 0.1 deg).
"""

import cdsapi

c = cdsapi.Client()

c.retrieve(
    "reanalysis-era5-land",
    {
        "variable": [
            "soil_temperature_level_1",
            "soil_temperature_level_2",
            "soil_temperature_level_3",
            "soil_temperature_level_4",
            "volumetric_soil_water_layer_1",
            "volumetric_soil_water_layer_2",
            "volumetric_soil_water_layer_3",
            "volumetric_soil_water_layer_4",
            "snow_depth_water_equivalent",
            "temperature_of_snow_layer",
            "snow_albedo",
            "snow_density",
            "skin_temperature",
        ],
        "year": "2004",
        "month": "01",
        "day": "01",
        "time": "00:00",
        "area": [53.55, 73.35, 18.15, 135.05],  # N, W, S, E
        "grid": [0.1, 0.1],
        "format": "netcdf",
    },
    "/data/yangjinhui/surf_pytorch/era5land/era5land_20040101_00.nc",
)
print("done")
