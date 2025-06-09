import pandas as pd
import numpy as np

# ----------------------------
# 1. データ読み込み
# ----------------------------
file_path = '../data/merged_data.csv'

df = pd.read_csv(file_path)

# ----------------------------
# 2. symmetry 列から情報抽出
# ----------------------------
pattern = (
    r"crystal_system=<CrystalSystem\.\w+: '(?P<crystal_system>[^']+)'>.*"
    r"symbol='(?P<symbol>[^']+)'.*"
    r"number=(?P<number>\d+).*"
    r"point_group='(?P<point_group>[^']+)'"
)
# named capture で一括抽出
df_sym = df['symmetry'].str.extract(pattern)
# number を整数型に変換
df_sym['number'] = df_sym['number'].astype(int)
# 元の df にマージ
df = pd.concat([df, df_sym], axis=1)

# ----------------------------
# 3. t-SNE 距離計算とソート
# ----------------------------
# 基準とする化学式を指定
target_formula = 'Ge3(Te3As)2'

# インデックス取得（先頭マッチを想定）
idx = df.index[df['formula_pretty'] == target_formula][0]
x0, y0 = df.loc[idx, ['tSNE_1', 'tSNE_2']]

# ユークリッド距離を計算
df['distance'] = np.sqrt((df['tSNE_1'] - x0)**2 + (df['tSNE_2'] - y0)**2)

# 距離が小さい順にソート
sorted_df = df.sort_values('distance')

# ----------------------------
# 4. 結果確認
# ----------------------------
# 抽出・距離列を含む上位10件を表示
print(sorted_df[
    ['formula_pretty', 'material_id', 'tSNE_1', 'tSNE_2',
     'cluster', 'distance', 'crystal_system', 'symbol', 'number', 'point_group']
].head(50))

output_file_path = '../result/analyzed_data.csv'
sorted_df.to_csv(output_file_path, index=False)