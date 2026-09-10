import os
import sys
import requests
import pandas as pd
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

# ค้นหาและโหลด .env จากทั้ง Working Directory และโฟลเดอร์ของไฟล์โปรแกรม
load_dotenv()
if getattr(sys, "frozen", False):
    load_dotenv(os.path.join(os.path.dirname(sys.executable), ".env"))
else:
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

def get_data():
    url = os.getenv("BASE_PATH_DATA_GO")
    resource_id = os.getenv("RESOURCE_ID")
    api_key = os.getenv("API_KEY")

    if not url:
        print("❌ ไม่พบค่า BASE_PATH_DATA_GO ใน .env")
        print("กรุณาตรวจสอบว่ามีไฟล์ .env วางอยู่ในโฟลเดอร์เดียวกับไฟล์โปรแกรมหรือไม่")
        return

    params = {
        'resource_id': resource_id,
        'limit': 100000
    }

    headers = {
        'api-key': api_key
    }

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