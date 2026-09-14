import os
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from preparation import get_preparation_data

df = get_preparation_data()

# เลือกตัวแปรสำหรับทำ cluster
feature = ['age', 'sex', 'risk']
X = df[feature].copy()

top_risks = X['risk'].value_counts().nlargest(5).index
X['risk'] = X['risk'].apply(lambda x: x if x in top_risks else 'อื่นๆ')

X_encoded = pd.get_dummies(X, columns=['sex', 'risk'], drop_first=False)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# ทำ k-means
k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)
print("จำนวนข้อมูลในแต่ละ Cluster:")
print(df['cluster'].value_counts())

# 1. ดูค่าเฉลี่ยของอายุในแต่ละกลุ่ม
print("\n--- อายุเฉลี่ยแยกตาม Cluster ---")
print(df.groupby('cluster')['age'].agg(['count', 'mean', 'median', 'min', 'max']))

# 2. ดูสัดส่วนเพศในแต่ละกลุ่ม
print("\n--- สัดส่วนเพศในแต่ละ Cluster ---")
print(pd.crosstab(df['cluster'], df['sex'], normalize='index') * 100)

# 3. ดูปัจจัยเสี่ยงหลัก (Risk) ในแต่ละกลุ่ม
print("\n--- ปัจจัยเสี่ยงหลักในแต่ละ Cluster ---")
print(pd.crosstab(df['cluster'], df['risk'], normalize='index') * 100)

# บันทึกตารางสรุปผล
os.makedirs('results', exist_ok=True)
try:
    plt.rcParams['font.family'] = 'Sarabun'
except Exception:
    pass

cluster_counts = df['cluster'].value_counts().sort_index()
total_patients = len(df)

table_data = []
for c_idx, count in cluster_counts.items():
    pct = (count / total_patients) * 100
    table_data.append([f"Cluster {c_idx}", f"{count:,}", f"{pct:.2f}%"])

table_data.append(["รวมทั้งหมด", f"{total_patients:,}", "100.00%"])
columns = ['กลุ่ม (Cluster)', 'จำนวนผู้ป่วย (คน)', 'สัดส่วน (%)']

fig, ax = plt.subplots(figsize=(6.5, 2.8), dpi=300)
ax.axis('off')

table = ax.table(
    cellText=table_data,
    colLabels=columns,
    cellLoc='center',
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.15, 1.8)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('#CBD5E1')
    cell.set_linewidth(1)
    if row == 0:
        cell.set_facecolor('#1E3A8A')
        cell.set_text_props(color='white', weight='bold')
    elif row == len(table_data):
        cell.set_facecolor('#E2E8F0')
        cell.set_text_props(weight='bold', color='#0F172A')
    else:
        cell.set_facecolor('#FFFFFF' if row % 2 != 0 else '#F8FAFC')
        cell.set_text_props(color='#334155')

plt.title('ตารางแสดงจำนวนและสัดส่วนผู้ป่วยในแต่ละ Cluster (ข้อ 1.3)', fontsize=13, weight='bold', pad=12, color='#1E293B')
plt.savefig('results/cluster_distribution_table.png', bbox_inches='tight', dpi=300)
plt.close()
print("บันทึกรูปภาพตารางข้อ 1.3 ไว้ที่: results/cluster_distribution_table.png เรียบร้อยแล้ว")

# ตารางลักษณะเฉพาะของแต่ละ Cluster
interpret_columns = ['ตัวชี้วัด', 'Cluster 0 (ชาย)', 'Cluster 1 (หญิง)', 'Cluster 2 (ไม่ระบุ)']
interpret_data = [
    ['สัดส่วนเพศหลัก', 'ชาย 100.0%', 'หญิง 100.0%', 'ไม่ระบุเพศ 100.0%'],
    ['อายุเฉลี่ย (Mean)', f"{df[df['cluster']==0]['age'].mean():.2f} ปี", f"{df[df['cluster']==1]['age'].mean():.2f} ปี", f"{df[df['cluster']==2]['age'].mean():.2f} ปี"],
    ['อายุมัธยฐาน (Median)', f"{df[df['cluster']==0]['age'].median():.1f} ปี", f"{df[df['cluster']==1]['age'].median():.1f} ปี", f"{df[df['cluster']==2]['age'].median():.1f} ปี"],
    ['ช่วงอายุ (Min - Max)', f"{int(df[df['cluster']==0]['age'].min())} - {int(df[df['cluster']==0]['age'].max())} ปี", f"{int(df[df['cluster']==1]['age'].min())} - {int(df[df['cluster']==1]['age'].max())} ปี", f"{int(df[df['cluster']==2]['age'].min())} - {int(df[df['cluster']==2]['age'].max())} ปี"],
    ['ปัจจัยเสี่ยงอันดับ 1', 'สัมผัสใกล้ชิดผู้ป่วยยืนยัน\n(35.67%)', 'สัมผัสใกล้ชิดผู้ป่วยยืนยัน\n(36.66%)', 'ปัจจัยเสี่ยงอื่นๆ\n(75.83%)'],
    ['ปัจจัยเสี่ยงอันดับ 2', 'อยู่ระหว่างการสอบสวน\n(15.42%)', 'อยู่ระหว่างการสอบสวน\n(14.11%)', 'สัมผัสใกล้ชิดผู้ป่วยยืนยัน\n(12.08%)']
]

fig_int, ax_int = plt.subplots(figsize=(11.0, 4.5), dpi=300)
ax_int.axis('off')

int_table = ax_int.table(
    cellText=interpret_data,
    colLabels=interpret_columns,
    cellLoc='center',
    loc='center'
)

int_table.auto_set_font_size(False)
int_table.set_fontsize(11)
int_table.scale(1.15, 2.1)

for (row, col), cell in int_table.get_celld().items():
    cell.set_edgecolor('#CBD5E1')
    cell.set_linewidth(1)
    if row == 0:
        cell.set_facecolor('#1E3A8A')
        cell.set_text_props(color='white', weight='bold')
    else:
        cell.set_facecolor('#FFFFFF' if row % 2 != 0 else '#F8FAFC')
        cell.set_text_props(color='#334155')
        if col == 0:
            cell.set_text_props(weight='bold', color='#1E293B')

plt.title('ตารางสรุปลักษณะเฉพาะของแต่ละ Cluster', fontsize=13, weight='bold', pad=14, color='#1E293B')
plt.savefig('results/cluster_interpret_table.png', bbox_inches='tight', dpi=300)
plt.close()
print("บันทึกรูปภาพตารางข้อ 1.4 ไว้ที่: results/cluster_interpret_table.png เรียบร้อยแล้ว")
