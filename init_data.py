import requests
import pandas as pd
import os
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
import json

load_dotenv()

url = os.getenv("BASE_PATH_DATA_GO")
resource_id = os.getenv("RESOURCE_ID")
api_key = os.getenv("API_KEY")

params = {
    'resource_id': resource_id,
    'limit': 100000
}

headers = {
    'api-key': api_key
}

def get_data():
    try:
        result = requests.get(url, params=params, headers=headers)
        result.raise_for_status()
        data = result.json()
        if(data.get('success')):
            record = data['result']['records']
            df = pd.DataFrame(record)
            if not os.path.exists("data"):
                os.makedirs("data")
            df.to_csv("data/covid_data.csv", index=False, encoding='utf-8-sig')
            print(f"Saved {len(df)} records to data/covid_data.csv")
        else:
            print("API Error:", data)
    except requests.exceptions.RequestException as e:
        print(e)

get_data()