

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


# some judge information is missing, so we used wikipedia search to fill in these values manually
appointing_party_map = {
    'Susan Pamela Watters': 'Democratic',
    'Pamela Pepper': 'Democratic',
    'Daniel Dale Crabtree': 'Democratic',
    'Darrin Phillip Gayles': 'Democratic',
    'Rosemary MÃ¡rquez': 'Democratic',
    'Robert William """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Trey"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" Schroeder': 'Democratic',
    'Mary Hannah Lauck': 'Democratic',
    'Timothy Lloyd Brooks': 'Democratic',
    'Linda Vivienne Parker': 'Democratic',
    'Nancy Jo Rosenstenge': 'Democratic',
    'Sheryl H  Lipman': 'Democratic',
    'Amos Louis Mazzant': 'Democratic',
    'Leigh Martin May': 'Democratic',
    'Staci Michelle Yandle': 'Democratic',
    'Eleanor Louise Ross': 'Democratic',
    'Douglas Leroy Rayes': 'Democratic',
    'Matthew Frederick Leitman': 'Democratic',
    'Laurie Jill Michelson': 'Democratic',
    'Stephen Rogers Bough': 'Democratic',
    'Stanley Allen Bastian': 'Democratic',
    'Loretta Copeland Biggs': 'Democratic',
    'Ronnie Lee White': 'Democratic',
    'Robin Lee Rosenberg': 'Democratic',
    'Mark Howard Cohen': 'Democratic',
    'Wendy Beetlestone': 'Democratic',
    'Mark Gerald Mastroianni': 'Democratic',
    'George Jarrod Hazel': 'Democratic',
    'Steven Paul Logan': 'Democratic',
    'Madeline Elizabeth Cox Arleo': 'Democratic',
    'Gregory Neil Stivers': 'Democratic',
    'Gerald John """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Jerry"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" Pappert': 'Democratic',
    'Dale Alan Drozd': 'Democratic',
    'Travis Randall McDonough': 'Democratic',
    'Lawrence Joseph Vilardo': 'Democratic',
    'Leonard Terry Strand': 'Democratic',
    'Rebecca Goodgame Ebinger': 'Democratic',
    'Paula Xinis': 'Democratic',
    'Scott Lawrence Palk': 'Republican',
    'Donald Cecil Coggins': 'Republican',
    'Alan D Albright': 'Republican',
    'Terry Fitzgerald Moorer': 'Republican',
    '""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""C  J """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" Williams': 'Republican',
    'Thomas Shawn Kleeh': 'Republican',
    'Barry Weldon Ashe': 'Republican',
    'James Patrick """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""J  P """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" Hanlon': 'Republican',
    'Robert Earl Wier': 'Republican',
    'Jeremy Daniel Kernodle': 'Republican',
    'James Russell Sweeney': 'Republican',
    'Liles Clifton Burke': 'Republican',
    'Susan Marie Brnovich': 'Republican',
    'William Frederic Jung': 'Republican',
    'Theodore David Chuang': 'Democratic',
    'Mark A  Kearney': 'Democratic',
    'Nancy Jo Rosenstengel': 'Democratic',
    'James Donald Peterson': 'Democratic',
    'Pamela Lynn Reeves': 'Democratic',
    'Paul Gregory Byron': 'Democratic',
    'James Alan Soto': 'Democratic',
    'Joseph F  Leeson': 'Democratic',
    'Edward George Smith': 'Democratic',
    'Jon David Levy': 'Democratic',
    'Amit Priyavn Mehta': 'Democratic',
    'Joan Marie Azrack': 'Democratic',
    'Diane Joyce Humetewa': 'Democratic',
    'Elizabeth Kay Dillon': 'Democratic',
    'Randolph Daniel Moss': 'Democratic',
    'Gary Allen Feess': 'Democratic',
    'Ann Marie Donnelly': 'Democratic',
    'Robert Francis Rossiter': 'Democratic',
    'Holly Lou Teeter': 'Republican',
    'Mark Raymond Hornak': 'Democratic',
    'Jennifer Guerin Zipps': 'Democratic',
    'John McCarthy Roll': 'Republican',
    'Adalberto Jose Jordan': 'Democratic',
    'Nannette Jolivette Brown': 'Democratic',
    'Katherine Bolan Forrest': 'Democratic',
    'Jeremy Don Fogel': 'Democratic',
    'Dana Lewis Christensen': 'Democratic',
    'Paul William Grimm': 'Democratic',
    'Matthew William Brann': 'Democratic',
    'John Thomas Fowlkes': 'Democratic',
    'Jeffrey James Helmick': 'Democratic',
    'Brian Curtis Wimes': 'Democratic',
    'Terrence George Berg': 'Democratic',
    'John Melvin Gerrard': 'Democratic',
    'George Levi Russell': 'Democratic',
    'Gina Marie Groh': 'Democratic',
    'Stephanie Marie Rose': 'Democratic',
    'Michael Walter Fitzgerald': 'Democratic',
    'Kevin McNulty': 'Democratic',
    'Mary Geiger Lewis': 'Democratic',
    'Michael Peter Shea': 'Democratic',
    'Richard James Holwell': 'Republican',
    'Gonzalo Paul Curiel': 'Democratic',
    'Robert James """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Bob"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" Shelby': 'Democratic',
    'Malachy Edward Mannion': 'Democratic',
    'Madeline Hughes Haikala': 'Democratic',
    'Debra Marie Brown': 'Democratic',
    'Jennifer Anna Dorsey': 'Democratic',
    'Jeffrey L  Schmehl': 'Democratic',
    'Frank Paul Geraci': 'Democratic',
    'Andrew Patrick Gordon': 'Democratic',
    'Troy Lynne Nunley': 'Democratic',
    'Landya Boyer McCafferty': 'Democratic',
    'Sheri Polster Chappell': 'Democratic',
    'Rachelle """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Shelly"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" Lynne Deckert Dick': 'Democratic',
    'Vernon Speede Broderick': 'Democratic',
    'Valerie E  Caproni': 'Democratic'
}

missing_race_sex_party_map = {
    'Amy Mil Totenberg': ['White', 'Female', 'Democratic'],
    'Analisa Nadine Torres': ['Hispanic', 'Female', 'Democratic'],
    'Ann Louise Aiken': ['White', 'Female', 'Democratic'],
    'Avern Levin Cohn': ['White', 'Male', 'Democratic'],
    'Beth Bloom Stern': ['White', 'Female', 'Democratic'],
    'Brian Theadore """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Ted"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" Stewart': ['White', 'Male', 'Democratic'],
    'Callie Virginia Smith """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Ginny"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" Granade': ['White', 'Female', 'Republican'],
    'Christina Clair Reiss': ['White', 'Female', 'Democratic'],
    'Clyde Roger Vinson': ['White', 'Male', 'Republican'],
    'Colleen Kollar Kotelly': ['White', 'Female', 'Democratic'],
    'Daniel T  K  Hurley': ['White', 'Male', 'Democratic'],
    'David Ogden Nuffer': ['White', 'Male', 'Democratic'],
    'Dee Vance Benson': ['White', 'Male', 'Democratic'],
    'Gene Ellen Kreyche Pratter': ['White', 'Female', 'Republican'],
    'Gerald John """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Jerry"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" Pappert': ['White', 'Male', 'Democratic'],
    'Gregory Frederick Van Tatenhove': ['White', 'Male', 'Republican'],
    'Ivan L  R  Lemelle': ['Black', 'Male', 'Democratic'],
    'Jane Elizabeth Magnus Stinson': ['White', 'Female', 'Democratic'],
    'Jeffrey Uhlman Beaverstock': ['White', 'Male', 'Republican'],
    'Jon Ernest DeGuilio': ['White', 'Male', 'Democratic'],
    'Laurie Smith Camp': ['White', 'Female', 'Republican'],
    'Leslie Joyce Abrams': ['Black', 'Female', 'Democratic'],
    'Martha Alicia VÃ¡zquez': ["Hispanic", 'Female', 'Democratic'],
    'Martin Leach Cross Feldman': ['White', 'Male', 'Republican'],
    'Nitza Ileana QuiÃ±ones Alejandro': ['Hispanic', 'Female', 'Democratic'],
    'Patricia Minaldi': ['White', 'Female', 'Republican'],
    'Ronald Sing Wai Lew': ['Other', 'Male', 'Republican'],
    'Stewart Richard Dalzell': ['White', 'Male', 'Republican'],
    'Susan Yvonne Illston': ['White', 'Female', 'Democratic'],
    'Thomas E  Stagg': ['White', 'Male', 'Republican'],
    'Vince Girdhari Chhabria': ['White', 'Male', 'Democratic'],
    'John McCarthy Roll': ['White', 'Male', 'Republican'],
    'Adalberto Jose Jordan': ['Hispanic', 'Male', 'Democratic'],
    'Jeremy Don Fogel': ['White', 'Male', 'Democratic']
}

# apply the missing race, sex, and party values to the justfair_df
for judge, (race, sex, party) in missing_race_sex_party_map.items():
    justfair_df.loc[justfair_df["JUDGE_NAME"] == judge, "JUDGE_RACE"] = race
    justfair_df.loc[justfair_df["JUDGE_NAME"] == judge, "JUDGE_SEX"] = sex
    justfair_df.loc[justfair_df["JUDGE_NAME"] == judge, "APPOINTING_PARTY"] = party

# apply the missing appointing party values to the justfair_df
for judge, party in appointing_party_map.items():
    justfair_df.loc[justfair_df["JUDGE_NAME"] == judge, "APPOINTING_PARTY"] = party

justfair_df.to_csv(os.path.join(data_path, "JUSTFAIR_clean.csv"), index=False)