import os
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
from mlxtend.frequent_patterns import apriori, association_rules
from preparation import get_preparation_association_rule

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

# บันทึกตารางผลลัพธ์
os.makedirs('results', exist_ok=True)
try:
    plt.rcParams['font.family'] = 'Sarabun'
except Exception:
    pass

table_data = []
for idx, r in strong_rules.reset_index().iterrows():
    ant = ', '.join(list(r['antecedents']))
    con = ', '.join(list(r['consequents']))
    sup = f"{r['support']:.3f} ({r['support']*100:.1f}%)"
    conf = f"{r['confidence']*100:.1f}%"
    lift = f"{r['lift']:.3f}"
    table_data.append([str(idx + 1), ant, con, sup, conf, lift])

columns = ['ลำดับ', 'เหตุ (Antecedents)', 'ผล (Consequents)', 'Support', 'Confidence', 'Lift']

fig, ax = plt.subplots(figsize=(12.5, 6.2), dpi=300)
ax.axis('off')

table = ax.table(
    cellText=table_data,
    colLabels=columns,
    cellLoc='center',
    loc='center',
    colWidths=[0.07, 0.45, 0.16, 0.13, 0.11, 0.08]
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.9)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('#CBD5E1')
    cell.set_linewidth(1)
    if row == 0:
        cell.set_facecolor('#1E3A8A')
        cell.set_text_props(color='white', weight='bold', size=10.5)
    else:
        cell.set_facecolor('#FFFFFF' if row % 2 != 0 else '#F8FAFC')
        cell.set_text_props(color='#334155')
        if col in [0, 4]:
            cell.set_text_props(weight='bold', color='#1E293B')
        if col == 1:
            cell.set_text_props(ha='left')

plt.title('ตารางสรุป Strong Association Rules (Top 11 เรียงตาม Confidence มากไปน้อย)', fontsize=13, weight='bold', pad=14, color='#1E293B')
plt.savefig('results/association_rules_table.png', bbox_inches='tight', dpi=300)
plt.close()
print("บันทึกรูปภาพตารางไว้ที่: results/association_rules_table.png เรียบร้อยแล้ว")



