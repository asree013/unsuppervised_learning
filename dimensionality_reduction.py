import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from preparation import get_preparation_data
from sklearn.decomposition import PCA
# pyrefly: ignore [missing-import]
import numpy as np
import os
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

# data preparation
df = get_preparation_data()

# select feature
feature = ["age", "sex", "risk"]
X_subset = df[feature].copy()

# ควบรวมค่าปัจจัยเสี่ยงที่พบน้อยให้เป็น 'อื่นๆ' เพื่อไม่ให้มิติขยายมากเกินไป
top_risk = X_subset["risk"].value_counts().nlargest(5).index
X_subset["risk"] = X_subset["risk"].apply(
    lambda x: x if x in top_risk else "อื่นๆ"
)

# One-Hot Encoding สำหรับตัวแปรประเภทกลุ่ม
X_encoded = pd.get_dummies(
    X_subset, columns=["sex", "risk"], drop_first=False
)
feature_names = X_encoded.columns.tolist()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)
# Fueture Selection
y_target = (df["sex"] == "ชาย").astype(
    int
) 

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, y_target)

importance_df = pd.DataFrame(
    {"Feature": feature_names, "Importance": rf.feature_importances_}
).sort_values(by="Importance", ascending=False)

# คำนวณ PCA แบบเต็มมิติเพื่อดูสัดส่วน Variance
pca_full = PCA()
pca_full.fit(X_scaled)

explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# for i, (var, cum_var) in enumerate(
#     zip(explained_variance, cumulative_variance), 1
# ):
#     print(
#         f"PC{i}: Variance = {var * 100:.2f}%, Cumulative Variance = {cum_var * 100:.2f}%"
#     )

# หาจำนวน Principal Components ที่เก็บ Variance ได้อย่างน้อย 80%
n_components_optimal = np.argmax(cumulative_variance >= 0.80) + 1
# print(f"\n>> จำนวน Component ที่เหมาะสม (ครอบคลุม Variance >= 80%): {n_components_optimal} Components")

# แปลงข้อมูลเป็น Principal Components ตามจำนวนที่เลือก
pca_optimal = PCA(n_components=n_components_optimal)
X_pca = pca_optimal.fit_transform(X_scaled)

pca_columns = [f"PC{i+1}" for i in range(n_components_optimal)]
df_pca = pd.DataFrame(X_pca, columns=pca_columns)
print("\nตัวอย่างข้อมูลหลังลดมิติ (5 แถวแรก):")
print(df_pca.head())

# 4. การแสดงผลด้วยกราฟ (Visualizations สำหรับรายงาน)
os.makedirs("results", exist_ok=True)

# 4.1 กราฟ Feature Importance (Bar chart)
plt.figure(figsize=(10, 6))
# เรียงลำดับจากน้อยไปมากเพื่อให้ค่าที่มากที่สุดอยู่ด้านบนของกราฟแนวนอน
sorted_df = importance_df.sort_values(by="Importance", ascending=True)
plt.barh(sorted_df["Feature"], sorted_df["Importance"], color="#3b82f6")
plt.title("Feature Importance Ranking (from Most to Least)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig("results/feature_importance.png", dpi=300)
plt.close()


# 4.2 กราฟ Scree Plot & Cumulative Explained Variance
plt.figure(figsize=(9, 5))
components_range = range(1, len(explained_variance) + 1)
plt.bar(
    components_range,
    explained_variance,
    alpha=0.6,
    label="Individual Explained Variance",
)
plt.step(
    components_range,
    cumulative_variance,
    where="mid",
    label="Cumulative Explained Variance",
    color="red",
)
plt.axhline(
    y=0.8, color="grey", linestyle="--", label="80% Threshold Threshold"
)
plt.xlabel("Principal Component Index")
plt.ylabel("Explained Variance Ratio")
plt.title("Scree Plot & Cumulative Explained Variance")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("results/pca_scree_plot.png", dpi=300)
plt.close()
print("\nบันทึกกราฟสรุปผลไว้ที่โฟลเดอร์ results/ เรียบร้อยแล้ว")


