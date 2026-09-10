from preparation import get_preparation_association_rule
import pandas as pd
# pyrefly: ignore [missing-import]
from mlxtend.frequent_patterns import apriori, association_rules

df = get_preparation_association_rule()

top_risks = df['risk'].value_counts().nlargest(4).index
print(top_risks)
df['risk'] = df['risk'].apply(lambda x: x if x in top_risks else 'อื่นๆ')

features = ['age', 'sex', 'risk']
X_encoded = pd.get_dummies(df[features], prefix=['อายุ', 'เพศ', 'ความเสี่ยง'])

frequent_itemsets = apriori(X_encoded, min_support=0.03, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.50)

strong_rules = rules[rules['lift'] > 1.0].sort_values(by='confidence', ascending=False).head(15)

print("="*80)
print(f"สรุป Strong Association Rules (Top {len(strong_rules)} เรียงตาม Confidence มากไปน้อย)")
print("="*80)
for idx, r in strong_rules.reset_index().iterrows():
    antecedents = list(r['antecedents'])
    consequents = list(r['consequents'])
    print(f"กฎข้อที่ {idx+1}: {antecedents} => {consequents}")
    print(f"   Support: {r['support']:.3f} | Confidence: {r['confidence']:.3f} ({r['confidence']*100:.1f}%) | Lift: {r['lift']:.3f}\n")



