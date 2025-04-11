import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# データフォルダのパスを指定
data_folder = "../data/model/"
output_folder = '../result'

# ファイル名を辞書で定義
files = {
    "CGCNN": "bulk_data_bulk_data_C_CGCNN_demo_50_DR_output.csv",
    "GCN": "bulk_data_bulk_data_C_GCN_demo_50_DR_output.csv",
    "MEGNet": "bulk_data_bulk_data_C_MEGNet_demo_50_DR_output.csv",
    "MPNN": "bulk_data_bulk_data_C_MPNN_demo_50_DR_output.csv",
    "SchNet": "bulk_data_bulk_data_C_SchNet_demo_50_DR_output.csv"
}

# フルパスを作成しデータを読み込み
dataframes = {key: pd.read_csv(data_folder + filename) for key, filename in files.items()}

# 最近接距離を計算する関数
def calculate_nearest_neighbor_distances(df):
    tsne_coords = df[['tSNE_1', 'tSNE_2']].to_numpy()
    pairwise_distances = cdist(tsne_coords, tsne_coords, metric='euclidean')
    np.fill_diagonal(pairwise_distances, np.inf)
    nearest_distances = np.min(pairwise_distances, axis=1)
    return nearest_distances

# 各データセットの最近接距離を計算
nearest_distances = {key: calculate_nearest_neighbor_distances(df) for key, df in dataframes.items()}

# 非数値データを除去
cleaned_distances = {model_name: distances[np.isfinite(distances)] for model_name, distances in nearest_distances.items()}

# KDEプロットを作成
plt.figure(figsize=(12, 6))
for model_name, distances in cleaned_distances.items():
    sns.kdeplot(distances, label=model_name)

# # プロットの調整
# plt.title('Kernel Density Estimation of Nearest Neighbor Distances')
# plt.xlabel('Distance')
# plt.ylabel('Density')
# plt.xlim(0, 4)  # x軸の範囲を設定
# plt.legend(title='Model')
# plt.tight_layout()
#
# # プロットを表示
# plt.show()

# KDEプロットの数值データを保存
for model_name, distances in cleaned_distances.items():
    kde = sns.kdeplot(distances, label=model_name)
    x_data = kde.get_lines()[-1].get_xdata()
    y_data = kde.get_lines()[-1].get_ydata()
    output_df = pd.DataFrame({f"Distance_{model_name}": x_data, f"Density_{model_name}": y_data})
    output_df.to_csv(f"{output_folder}/kde_plot_data_{model_name}.csv", index=False)

# 計算した距離のデータをファイルに保存
# output_folder = "/path/to/save/output/"
for model_name, distances in cleaned_distances.items():
    output_df = pd.DataFrame({"Nearest_Distance": distances})
    output_df.to_csv(f"{output_folder}/nearest_distances_{model_name}.csv", index=False)
