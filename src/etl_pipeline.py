"""
Module: etl_pipeline.py
Loads and cleans synthetic clinical data for downstream NLP and modeling.
"""
import pandas as pd

def load_data(path="../data/synthetic_clinical_data.csv"):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    # Example cleaning: drop duplicates, handle missing values
    df = df.drop_duplicates()
    df = df.fillna("")
    return df

def etl_process(path="../data/synthetic_clinical_data.csv"):
    df = load_data(path)
    df = clean_data(df)
    return df

if __name__ == "__main__":
    df = etl_process()
    print(df.head())
