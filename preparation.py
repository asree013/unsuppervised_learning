import pandas as pd

def get_preparation_data() -> pd.DataFrame:
    df = pd.read_csv("data/covid_data.csv")
    # data preparation
    check_null_df = df.isnull().sum()
    text_cols = ['sex', 'nationality', 'province_of_onset', 'district_of_onset']
    df[text_cols] = df[text_cols].fillna("ไม่ระบุ")
    df['age'] = df['age'].fillna(df['age'].median())
    df["Unit"] = df['Unit'].fillna(df['Unit'].mode()[0])
    # print(df.isna().sum())
    return df

def get_preparation_association_rule() -> pd.DataFrame:
    df = pd.read_csv("data/covid_data.csv")
    # data preparation
    check_null_df = df.isnull().sum()
    text_cols = ['sex', 'nationality', 'province_of_onset', 'district_of_onset']
    df[text_cols] = df[text_cols].fillna("ไม่ระบุ")
    df['age'] = df['age'].fillna(df['age'].median())

    df["age"] = df["age"].apply(convert_age_to_enum)
    df["Unit"] = df['Unit'].fillna(df['Unit'].mode()[0])
    # print(df.isna().sum())
    return df

def convert_age_to_enum(age: int):
    if(age >= 60):
        return "ผู้สูงอายุ"
    elif(age >= 40 and age < 60):
        return "วัยกลางคน"
    elif(age >= 20 and age < 40):
        return "วัยทำงาน"
    elif(age >= 18 and age < 20):
        return "วัยรุ่น-คนรุ่นใหม่"
    elif(age >= 10 and age < 18):
        return "เด็ก"
    elif(age < 10):
        return "ทารก"
    