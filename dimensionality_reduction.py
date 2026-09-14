import os
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from preparation import get_preparation_data

# ตั้งค่าฟอนต์ภาษาไทย
for font_name in ["Sarabun", "Thonburi", "Tahoma", "Sukhumvit Set", "Ayuthaya", "DejaVu Sans"]:
    try:
        plt.rcParams["font.family"] = font_name
        break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False

print("=" * 70)
print("  การลดมิติข้อมูล (DIMENSIONALITY REDUCTION): FEATURE SELECTION & PCA")
print("=" * 70)

# เตรียมข้อมูล
print("\n[ขั้นตอนที่ 1] โหลดและเตรียมข้อมูล...")
df = get_preparation_data()

features = ["age", "sex", "risk"]
X_subset = df[features].copy()

top_risks = X_subset["risk"].value_counts().nlargest(5).index
X_subset["risk"] = X_subset["risk"].apply(lambda x: x if x in top_risks else "อื่นๆ")

X_encoded = pd.get_dummies(X_subset, columns=["sex", "risk"], drop_first=False)
feature_names = X_encoded.columns.tolist()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)
print(f"-> จำนวนตัวอย่าง (Samples): {X_scaled.shape[0]:,} รายการ")
print(f"-> จำนวนมิติข้อมูลตั้งต้นหลัง Encoding: {X_scaled.shape[1]} ฟีเจอร์")
print(f"-> รายชื่อฟีเจอร์: {', '.join(feature_names)}")

# Feature Selection (Random Forest)
print("\n" + "=" * 70)
print("[ขั้นตอนที่ 2] การประเมินความสำคัญของฟีเจอร์ (Feature Selection)")
print("=" * 70)

y_target = (df["sex"] == "ชาย").astype(int)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, y_target)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False).reset_index(drop=True)
importance_df["Rank"] = range(1, len(importance_df) + 1)
importance_df["Importance (%)"] = (importance_df["Importance"] * 100).round(2)

print("\nตารางจัดอันดับความสำคัญของฟีเจอร์ (Feature Importance Ranking จากมากไปน้อย):")
print("-" * 65)
print(f"{'ลำดับ':<6}{'ฟีเจอร์':<42}{'สัดส่วนสำคัญ (%)':<15}")
print("-" * 65)
for _, row in importance_df.iterrows():
    print(f" {row['Rank']:<5}{row['Feature']:<42}{row['Importance (%)']:>6.2f}%")
print("-" * 65)

# Feature Extraction (PCA)
print("\n" + "=" * 70)
print("[ขั้นตอนที่ 3] การสกัดฟีเจอร์ด้วย Principal Component Analysis (PCA)")
print("=" * 70)

pca_full = PCA()
pca_full.fit(X_scaled)

explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)
cov_matrix = np.cov(X_scaled.T)
eigenvalues = np.sort(np.linalg.eigvals(cov_matrix))[::-1].real

# เลือกจำนวน component ที่ variance สะสม >= 80%
threshold = 0.80
n_components_optimal = int(np.argmax(cumulative_variance >= threshold) + 1)

print("\nตารางแจกแจงความแปรปรวนของแต่ละ Principal Component:")
print("-" * 70)
print(f"{'Component':<12}{'Eigenvalue':<15}{'Variance (%)':<18}{'Cumulative (%)':<15}{'สถานะ':<10}")
print("-" * 70)
for i in range(len(explained_variance)):
    status = "เลือก (Optimal)" if i < n_components_optimal else "ตัดทิ้ง"
    print(f" PC{i+1:<9}{eigenvalues[i]:<15.3f}{explained_variance[i]*100:>6.2f}%{'':<10}{cumulative_variance[i]*100:>6.2f}%{'':<8}{status}")
print("-" * 70)

print(f"\n>> ผลการวิเคราะห์จำนวน Component ที่เหมาะสม (เกณฑ์ Variance สะสม >= {int(threshold*100)}%):")
print(f"   -> จำนวน Component ที่เลือก: {n_components_optimal} Components (PC1 ถึง PC{n_components_optimal})")
print(f"   -> ความแปรปรวนสะสมที่เก็บได้: {cumulative_variance[n_components_optimal - 1]*100:.2f}%")
print(f"   -> ลดขนาดมิติข้อมูลลงได้: {((len(explained_variance) - n_components_optimal) / len(explained_variance))*100:.1f}%")

pca_optimal = PCA(n_components=n_components_optimal)
X_pca = pca_optimal.fit_transform(X_scaled)
pca_columns = [f"PC{i+1}" for i in range(n_components_optimal)]
df_pca = pd.DataFrame(X_pca, columns=pca_columns)

print("\nตัวอย่างข้อมูลหลังการลดมิติด้วย PCA (5 แถวแรก):")
print(df_pca.head())

# คำนวณ loadings
loadings = pd.DataFrame(
    pca_optimal.components_[:2].T,
    columns=["PC1_Loading", "PC2_Loading"],
    index=feature_names
)
print("\nPCA Loadings สำหรับ 2 องค์ประกอบหลักแรก (PC1 และ PC2):")
print(loadings.round(3))

# บันทึกกราฟและตารางสรุปผล
os.makedirs("results", exist_ok=True)

# กราฟ Feature Importance
plt.figure(figsize=(10, 5.5), dpi=300)
sorted_df = importance_df.sort_values(by="Importance", ascending=True)
bars = plt.barh(sorted_df["Feature"], sorted_df["Importance"], color="#2563EB", edgecolor="#1D4ED8", height=0.65)
for bar in bars:
    w = bar.get_width()
    if w > 0.005:
        plt.text(w + 0.008, bar.get_y() + bar.get_height() / 2, f"{w*100:.2f}%", va="center", ha="left", fontsize=9, color="#1E293B")
plt.title("Feature Importance Ranking (จัดอันดับความสำคัญจากมากไปน้อย)", fontsize=13, weight="bold", pad=12, color="#0F172A")
plt.xlabel("Importance Score (ค่าความสำคัญ)", fontsize=11, labelpad=8)
plt.ylabel("Features (ฟีเจอร์)", fontsize=11)
plt.xlim(0, max(importance_df["Importance"]) * 1.15)
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("results/feature_importance.png", dpi=300)
plt.close()

# Scree Plot
plt.figure(figsize=(9.5, 5.2), dpi=300)
components_range = range(1, len(explained_variance) + 1)
plt.bar(
    components_range,
    explained_variance * 100,
    alpha=0.7,
    color="#3B82F6",
    edgecolor="#1D4ED8",
    label="Individual Explained Variance (%)",
)
plt.step(
    components_range,
    cumulative_variance * 100,
    where="mid",
    label="Cumulative Explained Variance (%)",
    color="#DC2626",
    linewidth=2.2,
    marker="o"
)
plt.axhline(
    y=threshold * 100,
    color="#059669",
    linestyle="--",
    linewidth=1.8,
    label=f"{int(threshold*100)}% Variance Threshold"
)
plt.axvline(
    x=n_components_optimal,
    color="#D97706",
    linestyle=":",
    linewidth=1.8,
    label=f"Optimal Components (k={n_components_optimal})"
)

for i, (var, cum) in enumerate(zip(explained_variance, cumulative_variance), 1):
    plt.text(i, var * 100 + 1.2, f"{var*100:.1f}%", ha="center", fontsize=8.5, color="#1E293B", weight="bold")

plt.xlabel("Principal Component Index (ลำดับแกนหลัก)", fontsize=11, labelpad=8)
plt.ylabel("Explained Variance Ratio (%)", fontsize=11, labelpad=8)
plt.title("PCA Scree Plot & Cumulative Explained Variance", fontsize=13, weight="bold", pad=12, color="#0F172A")
plt.xticks(components_range, [f"PC{i}" for i in components_range])
plt.ylim(0, 110)
plt.legend(loc="center right", frameon=True)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("results/pca_scree_plot.png", dpi=300)
plt.close()

# 2D PCA Scatter Plot
plt.figure(figsize=(8.5, 6), dpi=300)
sample_indices = np.random.RandomState(42).choice(len(df_pca), size=min(3000, len(df_pca)), replace=False)
sex_sample = df["sex"].iloc[sample_indices].values
colors = {"ชาย": "#2563EB", "หญิง": "#EC4899", "ไม่ระบุ": "#94A3B8"}

for sex_val, color in colors.items():
    mask = sex_sample == sex_val
    if np.any(mask):
        plt.scatter(
            df_pca.iloc[sample_indices].loc[mask, "PC1"],
            df_pca.iloc[sample_indices].loc[mask, "PC2"],
            alpha=0.5,
            s=22,
            c=color,
            label=f"เพศ: {sex_val}",
            edgecolors="none"
        )

plt.title("2D Projection using PCA (PC1 vs PC2)", fontsize=13, weight="bold", pad=12, color="#0F172A")
plt.xlabel(f"PC1 (Explained Variance: {explained_variance[0]*100:.2f}%)", fontsize=11, labelpad=8)
plt.ylabel(f"PC2 (Explained Variance: {explained_variance[1]*100:.2f}%)", fontsize=11, labelpad=8)
plt.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
plt.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
plt.legend(loc="upper right", frameon=True)
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("results/pca_2d_projection.png", dpi=300)
plt.close()

# ตารางสรุป PCA
table_data = []
for i in range(len(explained_variance)):
    status = f"เลือก (Optimal: ครอบคลุม >= {int(threshold*100)}%)" if i < n_components_optimal else "ตัดออก (Dimensionality Reduced)"
    table_data.append([
        f"PC{i+1}",
        f"{eigenvalues[i]:.3f}",
        f"{explained_variance[i]*100:.2f}%",
        f"{cumulative_variance[i]*100:.2f}%",
        status
    ])

summary_cols = ["Component", "Eigenvalue", "Variance (%)", "Cumulative (%)", "ผลการคัดเลือก"]
fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=300)
ax.axis("off")

summary_table = ax.table(
    cellText=table_data,
    colLabels=summary_cols,
    cellLoc="center",
    loc="center",
    colWidths=[0.13, 0.15, 0.16, 0.18, 0.38]
)
summary_table.auto_set_font_size(False)
summary_table.set_fontsize(10)
summary_table.scale(1.0, 1.8)

for (row, col), cell in summary_table.get_celld().items():
    cell.set_edgecolor("#CBD5E1")
    cell.set_linewidth(1)
    if row == 0:
        cell.set_facecolor("#1E3A8A")
        cell.set_text_props(color="white", weight="bold", size=10.5)
    elif row <= n_components_optimal:
        cell.set_facecolor("#EFF6FF" if row % 2 != 0 else "#DBEAFE")
        cell.set_text_props(color="#1E293B")
        if col == 4:
            cell.set_text_props(weight="bold", color="#1D4ED8")
    else:
        cell.set_facecolor("#F8FAFC" if row % 2 != 0 else "#FFFFFF")
        cell.set_text_props(color="#64748B")

plt.title("ตารางสรุปการลดมิติข้อมูลด้วย Principal Component Analysis (PCA)", fontsize=13, weight="bold", pad=14, color="#1E293B")
plt.savefig("results/pca_summary_table.png", bbox_inches="tight", dpi=300)
plt.close()

# ตารางสรุป loadings
loadings_interp_map = {
    "sex_หญิง": "แกนหลักบวกของ PC1 (บ่งบอกกลุ่มเพศหญิงอย่างชัดเจน)",
    "sex_ชาย": "แกนหลักลบของ PC1 (บ่งบอกกลุ่มเพศชายอย่างชัดเจน)",
    "risk_อื่นๆ": "แกนหลักบวกของ PC2 (กลุ่มผู้ป่วยที่มีปัจจัยเสี่ยงทั่วไป/อื่นๆ)",
    "risk_สัมผัสใกล้ชิดกับผู้ป่วยยืนยันรายก่อนหน้านี้": "แกนหลักลบของ PC2 (กลุ่มผู้ป่วยเสี่ยงสูงจากการสัมผัสใกล้ชิด)",
    "sex_ไม่ระบุ": "มีน้ำหนักปานกลางเชิงบวกใน PC2",
    "risk_อยู่ระหว่างการสอบสวน": "มีน้ำหนักเล็กน้อยเชิงลบใน PC2",
    "age": "น้ำหนักต่ำ (กระจายตัวอิสระ ไม่ครอบงำแกน PC1/PC2)",
    "risk_ระบุไม่ได้": "มีอิทธิพลน้อยมากต่อการจำแนกของทั้งสองแกน",
    "risk_การค้นหาผู้ป่วยเชิงรุกและค้นหาผู้ติดเชื้อในชุมชน": "มีอิทธิพลน้อยมากต่อการจำแนกของทั้งสองแกน"
}

sorted_features = sorted(
    feature_names,
    key=lambda f: max(abs(loadings.loc[f, "PC1_Loading"]), abs(loadings.loc[f, "PC2_Loading"])),
    reverse=True
)

loading_table_data = []
for idx, f in enumerate(sorted_features, 1):
    pc1_val = loadings.loc[f, "PC1_Loading"]
    pc2_val = loadings.loc[f, "PC2_Loading"]
    interp = loadings_interp_map.get(f, "ตัวแปรประกอบร่วมในการฉายมิติ")
    f_display = f if len(f) <= 35 else f[:32] + "..."
    loading_table_data.append([
        str(idx),
        f_display,
        f"{pc1_val:+.3f}",
        f"{pc2_val:+.3f}",
        interp
    ])

loading_cols = ["ลำดับ", "ฟีเจอร์ตั้งต้น (Original Feature)", "PC1 Loading", "PC2 Loading", "บทบาทและการแปลความหมาย (Interpretation)"]
fig, ax = plt.subplots(figsize=(12.8, 5.8), dpi=300)
ax.axis("off")

l_table = ax.table(
    cellText=loading_table_data,
    colLabels=loading_cols,
    cellLoc="center",
    loc="center",
    colWidths=[0.06, 0.32, 0.13, 0.13, 0.46]
)
l_table.auto_set_font_size(False)
l_table.set_fontsize(10)
l_table.scale(1.0, 1.85)

for (row, col), cell in l_table.get_celld().items():
    cell.set_edgecolor("#CBD5E1")
    cell.set_linewidth(1)
    if row == 0:
        cell.set_facecolor("#1E3A8A")
        cell.set_text_props(color="white", weight="bold", size=10.5)
    else:
        pc1_score = float(loading_table_data[row-1][2])
        pc2_score = float(loading_table_data[row-1][3])
        
        cell.set_facecolor("#FFFFFF" if row % 2 != 0 else "#F8FAFC")
        cell.set_text_props(color="#334155")

        if col == 2 and abs(pc1_score) >= 0.5:
            cell.set_facecolor("#DCFCE7" if pc1_score > 0 else "#FEE2E2")
            cell.set_text_props(weight="bold", color="#15803D" if pc1_score > 0 else "#B91C1C")
        elif col == 3 and abs(pc2_score) >= 0.5:
            cell.set_facecolor("#DCFCE7" if pc2_score > 0 else "#FEE2E2")
            cell.set_text_props(weight="bold", color="#15803D" if pc2_score > 0 else "#B91C1C")
            
        if col in [1, 4]:
            cell.set_text_props(ha="left")

plt.title("ตารางสรุปค่าน้ำหนักและการแปลความหมายของแกนหลัก (PCA Loadings Interpretation: PC1 & PC2)", fontsize=13, weight="bold", pad=14, color="#1E293B")
plt.savefig("results/pca_loadings_table.png", bbox_inches="tight", dpi=300)
plt.close()

# ตารางข้อมูลหลังลดมิติ 5 แถวแรก
trans_data = []
for row_idx, row_vals in df_pca.head().iterrows():
    trans_data.append([str(row_idx)] + [f"{v:.4f}" for v in row_vals])

trans_headers = ["Row Index"] + list(df_pca.columns)
fig, ax = plt.subplots(figsize=(9.5, 3.4), dpi=300)
ax.axis("off")

trans_table = ax.table(
    cellText=trans_data,
    colLabels=trans_headers,
    cellLoc="center",
    loc="center",
    colWidths=[0.14] * len(trans_headers)
)
trans_table.auto_set_font_size(False)
trans_table.set_fontsize(10.5)
trans_table.scale(1.0, 1.9)

for (row, col), cell in trans_table.get_celld().items():
    cell.set_edgecolor("#CBD5E1")
    cell.set_linewidth(1)
    if row == 0:
        cell.set_facecolor("#1E3A8A")
        cell.set_text_props(color="white", weight="bold", size=11)
    else:
        cell.set_facecolor("#FFFFFF" if row % 2 != 0 else "#F8FAFC")
        cell.set_text_props(color="#1E293B")
        if col == 0:
            cell.set_text_props(weight="bold", color="#475569")

plt.title("ตัวอย่างข้อมูลหลังการลดมิติด้วย PCA (Transformed Dataset: 5 แถวแรก)", fontsize=12.5, weight="bold", pad=14, color="#1E293B")
plt.savefig("results/pca_transformed_data_table.png", bbox_inches="tight", dpi=300)
plt.close()

print("\n" + "=" * 70)
print("บันทึกกราฟและตารางสรุปผลไว้ที่โฟลเดอร์ results/ ครบถ้วนแล้ว:")
print(" 1. results/feature_importance.png         (กราฟจัดอันดับ Feature Importance)")
print(" 2. results/pca_scree_plot.png              (กราฟ Scree Plot & Cumulative Variance)")
print(" 3. results/pca_2d_projection.png           (กราฟ 2D PCA Scatter Plot: PC1 vs PC2)")
print(" 4. results/pca_summary_table.png           (รูปภาพตารางสรุปการคัดเลือก Principal Components)")
print(" 5. results/pca_loadings_table.png          (รูปภาพตารางแปลความหมาย PCA Loadings: PC1 & PC2)")
print(" 6. results/pca_transformed_data_table.png  (รูปภาพตารางตัวอย่างข้อมูลหลังลดมิติ 5 แถวแรก)")
print("=" * 70 + "\n")


