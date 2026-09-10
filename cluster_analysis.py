from preparation import get_preparation_data
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


df = get_preparation_data()


#select variable for do cluster
feature = ['age', 'sex', 'risk']
X = df[feature].copy()

top_risks = X['risk'].value_counts().nlargest(5).index
X['risk'] = X['risk'].apply(lambda x: x if x in top_risks else 'อื่นๆ')

X_encoded = pd.get_dummies(X, columns=['sex', 'risk'], drop_first=False)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# do k-means
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
