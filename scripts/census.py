
"""
Utilities for fetching U.S. Census data via the Census Data API.
"""

import requests
import pandas as pd
from typing import List
from pathlib import Path
import os


def get_county_data_acs5(
        year: int, 
        api_key: str, 
        var_codes: List[str], 
        var_names: List[str], 
        timeout: int = 60
    ) -> pd.DataFrame:

    """
    Fetch data for every county via the U.S. Census Data API (ACS 5-year profile) for a given year.
    
    Args:
        year (int): The year of the ACS data to retrieve.
        api_key (str): Your U.S. Census API key.
        var_codes (List[str]): List of variable codes to retrieve. The variable codes can be found at https://api.census.gov/data/{year}/acs/acs5/profile/variables.html or https://api.census.gov/data/{year}/acs/acs5/subject/variables.html
        var_names (List[str]): List of desired variable names corresponding to the variable codes.
        timeout (int): Timeout for the API request in seconds. Default is 60 seconds.
    """

    # ucgid=pseudo(0100000US$0500000) gets all counties in the US
    base_url = "https://api.census.gov/data/{year}/acs/acs5/profile?get={vars}&ucgid=pseudo(0100000US$0500000)"
    api_call = base_url.format(
        year=year, 
        vars="NAME,"+",".join(var_codes), 
        api_key=api_key
    )

    r = requests.get(api_call, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    # Census sometimes can return {"error": "..."}
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(data["error"])
    
    df = pd.DataFrame(data[1:], columns=["NAME"]+var_names+["geoid"])
    return df

def get_county_data_all_yrs(
        start_yr: int, 
        end_yr: int, 
        api_key: str, 
        var_codes: List[str], 
        var_names: List[str],
        filename: str = None
    ) -> pd.DataFrame:
    """
    Fetch data for every county via the U.S. Census Data API (ACS 5-year profile) for a range of years.
    
    Args:
        start_yr (int): The starting year of the ACS data to retrieve.
        end_yr (int): The ending year of the ACS data to retrieve.
        api_key (str): Your U.S. Census API key.
        var_codes (List[str]): List of variable codes to retrieve. The variable codes can be found at https://api.census.gov/data/{year}/acs/acs5/profile/variables.html or https://api.census.gov/data/{year}/acs/acs5/subject/variables.html
        var_names (List[str]): List of desired variable names corresponding to the variable codes.
        filename (str): Optional. If provided, saves the combined DataFrame to this CSV file.
    """

    all_dfs = []
    for year in range(start_yr, end_yr + 1):
        df_year = get_county_data_acs5(year, api_key, var_codes, var_names)
        df_year["CEN_YR"] = year
        all_dfs.append(df_year)
    combined_df = pd.concat(all_dfs, ignore_index=True).sort_values(by=["geoid", "CEN_YR"])

    if filename:
        combined_df.to_csv(
            Path(__file__).parent.parent/"data"/filename, 
            index=False
        )
        print(f"Saved data to {filename}")
    return combined_df


if __name__ == "__main__":
    # test usage
    df = get_county_data_acs5(
        year=2021, 
        var_codes=["DP05_0001E"], 
        var_names=["Total Population"],
        api_key=None
    )
    print(df.head())
