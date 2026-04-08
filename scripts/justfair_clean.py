

"""
Cleans JUSTFAIR.csv and creates JUSTFAIR_clean.csv with only necessary columns for 
joining with sentencing data, and adds a column for the appointing party of the judge 
at the time of sentencing.
"""

# non built-in libraries
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# built-in libraries
import os
from pathlib import Path
import sys
import datetime
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# personally-defined modules
sys.path.append(os.path.join(str(Path.cwd()), "../"))  
from data_download import download_files_from_fld

# install reqd. datasets from Google Drive
folder_id = "1P2FRAkPrqL2nn2MNMyd4ilWbXNS_kkKD" 
data_path = os.path.join(str(Path.cwd()), "./data")
download_files_from_fld(folder_id, data_path)

# constants
START_DATE = datetime.datetime(2011, 10, 1)  # init start date of analysis to first day of FY 2012
END_DATE = datetime.datetime(2024, 9, 30)  # init end date of analysis to last day of FY 2024

# load raw sentencing data
sent_df = pd.read_csv(
    os.path.join(data_path, "opafy12_24_combined_filtered.csv"), 
    low_memory=False, 
    encoding='latin1'
)

# only keep columns necessary for joining
cols_to_include = [
    "SENTMON", "SENTYR", "CIRCDIST", "AGE", "CITWHERE", "EDUCATN", 
    "NEWRACE", "NEWCNVTN", "NUMDEPEN", "PRESENT", "SENTIMP", 
    "XCRHISSR", "XFOLSOR", "judge_clean_full",
    "BirthYear", "Gender", "RaceorEthnicity", "FISCALYR",
    "PartyofAppointingPresident1", "CommissionDate1", "TerminationDate1",
    "PartyofAppointingPresident2", "CommissionDate2", "TerminationDate2",
    "PartyofAppointingPresident3", "CommissionDate3", "TerminationDate3",
    "PartyofAppointingPresident4", "CommissionDate4", "TerminationDate4",
    "PartyofAppointingPresident5", "CommissionDate5", "TerminationDate5",
    "PartyofAppointingPresident6", "CommissionDate6", "TerminationDate6",
]

justfair_df = pd.read_csv(
    os.path.join(data_path, "JUSTFAIR.csv"), 
    low_memory=False, 
    usecols=cols_to_include
)
justfair_df = justfair_df[justfair_df["FISCALYR"]>=sent_df["YEAR"].min()]
justfair_df = justfair_df.rename(columns={c: c.upper() for c in justfair_df.columns})
justfair_df = justfair_df.rename(columns={
    "JUDGE_CLEAN_FULL": "JUDGE_NAME",
    "SENIOR": "JUDGE_SENIOR_STATUS",
    "BIRTHYEAR": "JUDGE_BIRTH_YEAR",
    "GENDER": "JUDGE_SEX",
    "RACEORETHNICITY": "JUDGE_RACE"
})

justfair_df["JUDGE_AGE"] = justfair_df["FISCALYR"] - justfair_df["JUDGE_BIRTH_YEAR"]
justfair_df = justfair_df.drop(columns=["JUDGE_BIRTH_YEAR"])

# make all commition date and termination date columns into datetime
for i in range(1, 7):
    justfair_df[f"COMMISSIONDATE{i}"] = pd.to_datetime(justfair_df[f"COMMISSIONDATE{i}"], errors="coerce")
    justfair_df[f"TERMINATIONDATE{i}"] = pd.to_datetime(justfair_df[f"TERMINATIONDATE{i}"], errors="coerce")


def get_party_for_spell(row):
    sent_date = row["SENT_DATE"]

    if pd.isna(sent_date):
        return pd.NA

    for i in range(1, 7):
        commission = row[f"COMMISSIONDATE{i}"]
        termination = row[f"TERMINATIONDATE{i}"]
        party = row[f"PARTYOFAPPOINTINGPRESIDENT{i}"]

        # skip if no commission date
        if pd.isna(commission):
            continue

        # still serving if termination is missing
        if sent_date >= commission and (pd.isna(termination) or sent_date <= termination):
            return party

    return pd.NA

justfair_df["SENT_DATE"] = pd.to_datetime(
    dict(year=justfair_df["SENTYR"], month=justfair_df["SENTMON"], day=1),
    errors="coerce"
)

justfair_df["APPOINTING_PARTY"] = justfair_df.apply(get_party_for_spell, axis=1)
justfair_df = justfair_df.drop(columns=sum([[f"COMMISSIONDATE{i}", f"TERMINATIONDATE{i}", f"PARTYOFAPPOINTINGPRESIDENT{i}"] for i in range(1, 7)], []))
justfair_df = justfair_df.drop(columns=["SENT_DATE", "FISCALYR"])
justfair_df.to_csv(os.path.join(data_path, "JUSTFAIR_clean.csv"), index=False)