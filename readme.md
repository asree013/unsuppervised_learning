# Unsupervised Learning & Data Mining Project
การประยุกต์ใช้การเรียนรู้แบบไม่มีผู้สอนและการทำเหมืองข้อมูล (Unsupervised Learning & Data Mining) บนชุดข้อมูลผู้ป่วย COVID-19 จากประเทศไทย

---

## 📌 แหล่งที่มาของข้อมูล (Data Source)
* **แหล่งข้อมูล:** สำนักงานพัฒนารัฐบาลดิจิทัล (องค์การมหาชน) ผ่าน [data.go.th](https://data.go.th)
* **หน่วยงานเจ้าของข้อมูล:** กรมควบคุมโรค กระทรวงสาธารณสุข
* **ลักษณะข้อมูล:** ข้อมูลผู้ติดเชื้อ COVID-19 รายบุคคล ประกอบด้วย เพศ (`sex`), อายุ (`age`), จังหวัดที่ตรวจพบ (`province_of_onset`), ปัจจัยเสี่ยง (`risk`) ฯลฯ

---

## 📂 โครงสร้างโปรเจกต์ (Project Structure)

```text
├── .github/workflows/
│   └── build.yml               # GitHub Actions สำหรับ Build .exe (Windows) และ macOS อัตโนมัติ
├── data/
│   └── covid_data.csv          # ไฟล์ข้อมูลที่ดาวน์โหลดมาจาก API (สร้างอัตโนมัติ)
├── results/                    # โฟลเดอร์เก็บกราฟสรุปผล (สร้างอัตโนมัติ)
│   ├── feature_importance.png  # กราฟจัดอันดับความสำคัญของฟีเจอร์
│   └── pca_scree_plot.png      # กราฟ Scree Plot & Cumulative Explained Variance
├── .env.example                # ตัวอย่างการกำหนดค่า Environment Variables
├── init_data.py                # สคริปต์ดาวน์โหลดข้อมูลจาก Open Data API
├── preparation.py             # ขั้นตอนเตรียมข้อมูล (Data Preparation / Cleaning)
├── association_rule.py         # โมเดลหากฎความสัมพันธ์ (Association Rule Mining - Apriori)
├── cluster_analysis.py         # โมเดลจัดกลุ่มข้อมูล (Cluster Analysis - K-Means)
├── dimensionality_reduction.py # การลดมิติข้อมูล (Feature Selection & PCA)
├── main.py                     # โปรแกรมหลัก (Interactive Console Menu)
├── requirements.txt            # รายการ Dependencies ที่ต้องติดตั้ง
└── README.md                   # คู่มือการใช้งานโปรเจกต์
```

---

## 🚀 ฟังก์ชันและหัวข้อการวิเคราะห์ในโปรเจกต์

### 1. Data Preparation (`preparation.py`)
* จัดการ Missing Values (เช่น เติมค่า Median ให้กับอายุ และเติมข้อความ 'ไม่ระบุ' ให้ตัวแปรกลุ่ม)
* จัดกลุ่มช่วงอายุ (`convert_age_to_enum`) สำหรับงาน Association Rules
* ทำ One-Hot Encoding และ Standard Scaling สำหรับการคำนวณทางคณิตศาสตร์

### 2. Association Rule Mining (`association_rule.py`)
* ใช้อัลกอริทึม **Apriori** ค้นหากฎความสัมพันธ์ระหว่าง ช่วงอายุ, เพศ, และปัจจัยเสี่ยง
* คัดกรองเฉพาะ Strong Rules ที่มีค่า Lift > 1.0 และเรียงลำดับตาม Confidence

### 3. Cluster Analysis (`cluster_analysis.py`)
* จัดกลุ่มผู้ป่วยด้วย **K-Means Clustering** ($k = 3$)
* สรุปสถิติเชิงลึกของแต่ละกลุ่ม (อายุเฉลี่ย, สัดส่วนเพศ, ปัจจัยเสี่ยงหลัก)

### 4. Dimensionality Reduction (`dimensionality_reduction.py`)
* **Feature Selection:** คำนวณและจัดอันดับความสำคัญของฟีเจอร์ (Feature Importance) ด้วย Random Forest เรียงลำดับจากมากไปน้อย พร้อมบันทึกกราฟแท่ง
* **Feature Extraction (PCA):** คำนวณ Principal Components, วิเคราะห์ Explained Variance Ratio เพื่อเลือกจำนวน Components ที่เหมาะสม (ครอบคลุม Variance สะสม $\ge 80\%$) พร้อมสร้าง Scree Plot

---

## ⚙️ การติดตั้งและการใช้งาน (Installation & Usage)

### 1. เตรียม Environment และติดตั้งไลบรารี
```bash
# สร้างและเปิดใช้งาน Virtual Environment
python3 -m venv .venv
source .venv/bin/activate       # บน macOS / Linux
# .venv\Scripts\activate        # บน Windows

# ติดตั้งแพ็กเกจที่จำเป็น
pip install -r requirements.txt
```

### 2. ตั้งค่าไฟล์ `.env`
สร้างไฟล์ `.env` ไว้ที่โฟลเดอร์หลักของโปรเจกต์ (หรือดูตัวอย่างจาก `.env.example`):
```env
BASE_PATH_DATA_GO=https://opendata.data.go.th/api/3/action/datastore_search
RESOURCE_ID=your_resource_id_here
API_KEY=your_api_key_here
```

### 3. รันโปรแกรมผ่าน Python
```bash
python main.py
```
*(ระบบจะตรวจสอบและดาวน์โหลดข้อมูลผ่าน `init_data.py` ในครั้งแรก และเปิดเมนู Console ให้เลือกวิเคราะห์ข้อมูล)*

---

## 📦 การแปลงเป็นไฟล์ Executable (.exe / macOS App) ด้วย PyInstaller

สามารถแปลงเป็นไฟล์โปรแกรมสำเร็จรูปไฟล์เดียว (Standalone Executable) ที่ดับเบิลคลิกรันได้ทันที:

```bash
pyinstaller --clean --onefile --console \
  --hidden-import=init_data \
  --hidden-import=preparation \
  --hidden-import=association_rule \
  --hidden-import=cluster_analysis \
  --hidden-import=dimensionality_reduction \
  --name "DataMiningApp" main.py
```
ไฟล์โปรแกรมที่ได้จะอยู่ในโฟลเดอร์ `dist/`

> **หมายเหตุสำคัญ:** ให้นำไฟล์ `.env` และโฟลเดอร์ `data/` ไปวางคู่กับไฟล์ Executable เสมอ เพื่อให้โปรแกรมสามารถเข้าถึงข้อมูลได้

---

## 🌐 การ Build ข้ามแพลตฟอร์มด้วย GitHub Actions (CI/CD)

โปรเจกต์นี้มีระบบ **GitHub Actions** ในตัว (`.github/workflows/build.yml`)
* เมื่อ `git push` ขึ้น GitHub ระบบจะเปิดเครื่อง **Windows** และ **macOS** บนคลาวด์เพื่อ Build ไฟล์:
  * **`DataMiningApp-Windows`** (ไฟล์ `DataMiningApp.exe` สำหรับเครื่อง Windows)
  * **`DataMiningApp-macOS`** (ไฟล์ Executable สำหรับเครื่อง Mac)
* สามารถเข้าไปดาวน์โหลดไฟล์ทั้งสองระบบได้ฟรีทันทีที่แท็บ **Actions > Build Executables > Artifacts** บน GitHub