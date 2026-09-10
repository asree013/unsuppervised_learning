import os
import sys
import runpy

# กำหนด Path ให้ระบบหาโมดูลและไฟล์ข้อมูลเจอเสมอ ไม่ว่าจะรันผ่าน python หรือ PyInstaller
if getattr(sys, "frozen", False):
    base_dir = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    app_dir = os.path.dirname(sys.executable)
else:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    app_dir = base_dir

# ย้าย Working Directory มาที่โฟลเดอร์ของตัวโปรแกรม (แก้ปัญหาดับเบิลคลิกจาก Finder/Explorer แล้วหา .env หรือ data ไม่เจอ)
try:
    os.chdir(app_dir)
except Exception:
    pass

for path in [base_dir, app_dir, os.getcwd()]:
    if path not in sys.path:
        sys.path.insert(0, path)


try:
    # pyrefly: ignore [missing-import]
    from dotenv import load_dotenv
    load_dotenv(os.path.join(app_dir, ".env"))
    load_dotenv() # fallback
except Exception:
    pass

# ระบุให้ PyInstaller วิเคราะห์และรวมโมดูลเหล่านี้เข้าสู่ไฟล์ .exe / .app อัตโนมัติ
if False:
    import init_data
    import preparation
    import association_rule
    import cluster_analysis
    import dimensionality_reduction

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

def print_banner():
    print("=" * 65)
    print("      UNSUPERVISED LEARNING & DATA MINING PROJECT")
    print("=" * 65)

def run_module(module_name: str):
    """
    ฟังก์ชันสำหรับรันโค้ดของแต่ละโมดูลอย่างปลอดภัย
    ใช้ runpy เพื่อให้รันใหม่ได้ทุกครั้งโดยไม่มีปัญหา Module Caching
    """
    print(f"\n[กำลังเริ่มประมวลผล {module_name} ...]\n")
    try:
        runpy.run_module(module_name, run_name="__main__")
    except Exception as e:
        print(f"\nเกิดข้อผิดพลาดในการรัน {module_name}: {e}")

def initialize_dataset():
    """
    ทำงาน init_data.py เพื่อตรวจสอบและเตรียมข้อมูล covid_data.csv
    """
    print("\n--- ขั้นตอน: ตรวจสอบและเตรียมชุดข้อมูล (init_data.py) ---")
    data_path = os.path.join("data", "covid_data.csv")
    
    if os.path.exists(data_path):
        print(f"พบไฟล์ข้อมูลเดิมอยู่ที่: {data_path}")
        user_choice = input("ต้องการดาวน์โหลดข้อมูลใหม่จาก API หรือไม่? (y/N): ").strip().lower()
        if user_choice == "y":
            run_module("init_data")
        else:
            print("ใช้ไฟล์ข้อมูลเดิมในการประมวลผลต่อ")
    else:
        print("ยังไม่พบไฟล์ข้อมูลในระบบ กำลังดาวน์โหลดข้อมูลจาก data.go.th ...")
        run_module("init_data")

def main():
    clear_screen()
    print_banner()

    # 1. รันส่วน init_data ก่อนเสมอตามที่ระบุ
    initialize_dataset()
    input("\nกด Enter เพื่อเข้าสู่เมนูหลัก...")

    # 2. เมนูให้ผู้ใช้เลือกการทำงาน
    while True:
        clear_screen()
        print_banner()
        print("กรุณาเลือกโมเดล/หัวข้อที่ต้องการทำงาน:")
        print("  [1] Association Rule Mining     (association_rule.py)")
        print("  [2] Cluster Analysis (K-Means)  (cluster_analysis.py)")
        print("  [3] Dimensionality Reduction    (dimensionality_reduction.py)")
        print("  [4] โหลดข้อมูลใหม่อีกครั้ง       (init_data.py)")
        print("  [0] ออกจากโปรแกรม (Exit)")
        print("-" * 65)

        choice = input("ใส่หมายเลขเมนู (0-4): ").strip()

        if choice == "1":
            clear_screen()
            print("=" * 65)
            print("  เมนูที่ 1: ASSOCIATION RULE MINING")
            print("=" * 65)
            run_module("association_rule")
            input("\nกด Enter เพื่อกลับสู่เมนูหลัก...")

        elif choice == "2":
            clear_screen()
            print("=" * 65)
            print("  เมนูที่ 2: CLUSTER ANALYSIS (K-MEANS)")
            print("=" * 65)
            run_module("cluster_analysis")
            input("\nกด Enter เพื่อกลับสู่เมนูหลัก...")

        elif choice == "3":
            clear_screen()
            print("=" * 65)
            print("  เมนูที่ 3: DIMENSIONALITY REDUCTION (FEATURE SELECTION & PCA)")
            print("=" * 65)
            run_module("dimensionality_reduction")
            input("\nกด Enter เพื่อกลับสู่เมนูหลัก...")

        elif choice == "4":
            clear_screen()
            run_module("init_data")
            input("\nกด Enter เพื่อกลับสู่เมนูหลัก...")

        elif choice == "0":
            print("\nปิดโปรแกรมเรียบร้อย ขอบคุณครับ")
            break
        else:
            input("\nหมายเลขเมนูไม่ถูกต้อง กรุณากด Enter แล้วเลือกใหม่อีกครั้ง...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nยกเลิกการทำงานโดยผู้ใช้")
    except Exception as e:
        print(f"\nเกิดข้อผิดพลาดที่ไม่คาดคิด: {e}")
        input("\nกด Enter เพื่อปิดโปรแกรม...")
