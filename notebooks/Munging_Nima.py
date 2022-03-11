import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("protests.csv")

# Creating dummy variables for state response.

state_response = ["stateresponse1", "stateresponse2", "stateresponse3",
                  "stateresponse4", "stateresponse5", "stateresponse6", "stateresponse7"]

df = pd.get_dummies(
    data=df, prefix=["1", "2", "3", "4", "5", "6", "7"], columns=state_response)

df["accomodation"] = df['1_accomodation'] + df['2_accomodation'] + df['3_accomodation'] + \
    df['4_accomodation'] + df['5_accomodation'] + \
    df['6_accomodation'] + df['7_accomodation']

df["arrests"] = df['1_arrests'] + df['2_arrests'] + df['3_arrests'] + \
    df['4_arrests'] + df['5_arrests'] + df['6_arrests'] + df['7_arrests']

df["beatings"] = df['1_beatings'] + df['2_beatings'] + df['3_beatings'] + \
    df['4_beatings'] + df['5_beatings'] + df['6_beatings'] + df['7_beatings']

df["crowd_dispersal"] = df['1_crowd dispersal'] + df['2_crowd dispersal'] + \
    df['3_crowd dispersal'] + df['4_crowd dispersal'] + \
    df['5_crowd dispersal'] + df['6_crowd dispersal']

df["ignore"] = df['1_ignore'] + df['2_ignore'] + df['3_ignore']

df["killings"] = df['1_killings'] + df['2_killings'] + df['3_killings'] + \
    df['4_killings'] + df['5_killings'] + df['6_killings'] + df['7_killings']

df["shootings"] = df['1_shootings'] + df['2_shootings'] + \
    df['3_shootings'] + df['4_shootings'] + df['5_shootings']


# Creating dummy variables for protesters' demands.
demands = ['protesterdemand1', 'protesterdemand2',
           'protesterdemand3', 'protesterdemand4']

df = pd.get_dummies(data=df, prefix=["1", "2", "3", "4"], columns=demands)

df["labor_wage_dispute"] = df['1_labor wage dispute'] + \
    df['2_labor wage dispute'] + \
    df['3_labor wage dispute'] + df['4_labor wage dispute']

df["land_farm_issue"] = df['1_land farm issue'] + \
    df['2_land farm issue'] + df['3_land farm issue'] + df['4_land farm issue']

df["police_brutality"] = df['1_police brutality'] + \
    df['2_police brutality'] + df['3_police brutality'] + df['4_police brutality']

df["political_behavior_or_process"] = df['1_political behavior, process'] + \
    df['2_political behavior, process'] + \
    df['3_political behavior, process'] + df['4_political behavior, process']

df["price_hike_or_tax_policy"] = df['1_price increases, tax policy'] + df['2_price increases, tax policy'] + \
    df['3_price increases, tax policy'] + df['4_price increases, tax policy']

df["removal_of_politician"] = df['1_removal of politician'] + \
    df['2_removal of politician'] + \
    df['3_removal of politician'] + df['4_removal of politician']

df["social_restrictions"] = df['1_social restrictions'] + \
    df['2_social restrictions'] + df['3_social restrictions']

# drop the columns that are not needed.
df = df.drop(['1_accomodation', '2_accomodation', '3_accomodation', '4_accomodation', '5_accomodation', '6_accomodation', '7_accomodation', '1_arrests', '2_arrests', '3_arrests', '4_arrests', '5_arrests', '6_arrests', '7_arrests', '1_beatings', '2_beatings', '3_beatings', '4_beatings', '5_beatings', '6_beatings', '7_beatings', '1_crowd dispersal', '2_crowd dispersal', '3_crowd dispersal', '4_crowd dispersal', '5_crowd dispersal', '6_crowd dispersal', '1_ignore', '2_ignore', '3_ignore', '1_killings', '2_killings', '3_killings', '4_killings', '5_killings', '6_killings', '7_killings', '1_shootings', '2_shootings', '3_shootings', '4_shootings', '5_shootings', '1_labor wage dispute', '2_labor wage dispute',
             '3_labor wage dispute', '4_labor wage dispute', '1_land farm issue', '2_land farm issue', '3_land farm issue', '4_land farm issue', '1_police brutality', '2_police brutality', '3_police brutality', '4_police brutality', '1_political behavior, process', '2_political behavior, process', '3_political behavior, process', '4_political behavior, process', '1_price increases, tax policy', '2_price increases, tax policy', '3_price increases, tax policy', '4_price increases, tax policy', '1_removal of politician', '2_removal of politician', '3_removal of politician', '4_removal of politician', '1_social restrictions', '2_social restrictions', '3_social restrictions', '4_.', '5_.', '7_.'], axis=1)

# The column startday has its value recorded with a decimal. Remove the decimals from the values of these column.
df['New_day'] = df['startday'].astype(str).str.split('.').str[0]
df['New_month'] = df['startmonth'].astype(str).str.split('.').str[0]
df['New_year'] = df['startyear'].astype(str).str.split('.').str[0]


# Convert 'New_day', 'New_month', and 'New_year' to pandas date-time format.
df['New_day'] = pd.to_datetime(df['New_day'], format='%Y%m%d', errors='ignore')
df['New_month'] = pd.to_datetime(
    df['New_month'], format='%Y%m%d', errors='ignore')
df['New_year'] = pd.to_datetime(
    df['New_year'], format='%Y%m%d', errors='ignore')

# Create a new column called 'Start_date' that bundles df['New_day'], df['New_month'], and df['New_year'] in one column.
df['Start_date'] = df[['New_day', 'New_month', 'New_year']].apply(
    lambda x: '-'.join(x), axis=1)
df['Start_date'] = pd.to_datetime(
    df['Start_date'], format='%Y-%m-%d', errors='ignore')

# Repeat the process for the enddate
df['End_day'] = df['endday'].astype(str).str.split('.').str[0]
df['End_month'] = df['endmonth'].astype(str).str.split('.').str[0]
df['End_year'] = df['endyear'].astype(str).str.split('.').str[0]

df['End_day'] = pd.to_datetime(df['End_day'], format='%Y%m%d', errors='ignore')
df['End_month'] = pd.to_datetime(
    df['End_month'], format='%Y%m%d', errors='ignore')
df['End_year'] = pd.to_datetime(
    df['End_year'], format='%Y%m%d', errors='ignore')

df['End_date'] = df[['End_day', 'End_month', 'End_year']].apply(
    lambda x: '-'.join(x), axis=1)
df['End_date'] = pd.to_datetime(
    df['End_date'], format='%Y-%m-%d', errors='ignore')

# Ensure that there is no unsupported operand type(s) for -: 'str' and 'str' for 'Start_date' and 'End_date'
df['Start_date'] = pd.to_datetime(df['Start_date'], errors='coerce')
df['End_date'] = pd.to_datetime(df['End_date'], errors='coerce')

df['protest_length'] = (df['Start_date'] - df['End_date']).dt.days

# some of our timedelta values are negative. The follwoing code will fix this by retrieving the absolute value of the timedelta.
df['protest_length'] = df['protest_length'].abs()
# the last step is to iterate through the column protest_length and change the values 0 to 1.
df['protest_length'] = df['protest_length'].replace(0, 1)

# Now we can get rid of the time-related columns that are no longer needed.
df.drop(['New_day', 'New_month', 'New_year', 'startday', 'startmonth',
        'startyear', 'endday', 'endmonth', 'endyear', 'Start_date', 'End_date', 'End_day', 'End_month', 'End_year'], axis=1, inplace=True)


def parse_texts(x):
    x = x.lower()

    if x == "dozens":
        return 50
    elif x == "hundreds":
        return 500
    elif x == "thousands":
        return 5000
    elif x == "tens of thousands":
        return 50000
    elif "hundreds of thousands" in x:
        return 250000
    elif "millions" in x:
        return 2000000
    elif "million" in x:
        return 1000000

    elif "about " in x:
        return x[6:]
    elif "more than " in x:
        return x[10:]

    elif "several" in x:
        if "dozen" in x:
            return 50
        elif "hundred" in x:
            return 500
        elif "thousand" in x:
            return 5000

    elif "hundreds" in x:
        return 500
    elif "thousands" in x:
        return 5000

    else:
        return x


def strip_chars(x):
    banned_chars = "+s><,"
    x = "".join([c for c in x if c not in banned_chars])

    try:
        x = int(x)
    finally:
        return x


def avg_hyphen(x):
    accepted_chars = "1234567890-"
    ind = 0

    x = "".join([c for c in x if c in accepted_chars])

    for i in range(len(x)):
        if x[i] == "-":
            ind = i

    lower = x[:ind]
    upper = x[ind+1:]

    if (lower == "") or (upper == ""):
        return np.nan

    return (int(lower) + int(upper)) / 2


def map_participants(x):
    while type(x) == str:
        x = parse_texts(x)
        if type(x) == str:
            x = strip_chars(x)
        if type(x) == str:
            x = avg_hyphen(x)
        if type(x) == str:
            x = np.nan
    return x


df["participants"] = df["participants"].map(map_participants)

df.dropna(subset=["participants"], inplace=True)
df = df[df["region"] != "Oceania"]

# drop "notes", "sources", 'id', 'ccode', "protestnumber", "protesteridentity", "participants_category", 'location', 'participants_category'.
df.drop(['notes', 'sources', 'id', 'ccode', "protestnumber", "protesteridentity",
        "participants_category", 'location'], axis=1, inplace=True)

df.isnull().sum()
# print(df.shape)

# check the unique valaues of all variables in the data frame.
unique = df.nunique()
unique

# a close look at the unique values show that some of dummy variables ('accomodation','arrests',;;'beatings','crowd_dispersal','ignore', 'killings' and 'removal_of_politician') have more than 2 values (0 or 1). We need to Write a code that fixes this problem by changing the values of the spotted columns to 1 if the value is greater than 0 and 0 otherwise.

for col in ['accomodation', 'arrests', 'beatings', 'crowd_dispersal', 'ignore', 'killings', 'removal_of_politician']:
    df[col] = df[col].apply(lambda x: 1 if x > 0 else 0)

# check the unique valaues again to ensure the code worked.
unique = df.nunique()
unique

df.info()

# create a chart 'protest_type_count_per_country_per_year' that shows the values of the number of protests per country per year.
protest_type_count_per_country_per_year = df.groupby