import os
import pandas as pd
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


sns.set_palette("rainbow", 30)


def load_data(folder_path, model_name):
    input_file_path = os.path.join(folder_path, f'bulk_data_bulk_data_C_{model_name}_demo_50_DR_output.csv')
    custom_columns = ['index', 'target', 'PCA_1', 'PCA_2', 'tSNE_1', 'tSNE_2']
    df = pd.read_csv(input_file_path, skiprows=1, names=custom_columns)
    dfA = df[['target', 'tSNE_1', 'tSNE_2']]
    return dfA


def perform_clustering(dfA, n_clusters=30):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    dfA['cluster'] = kmeans.fit_predict(dfA)
    return dfA


def merge_data(dfA):
    folder_path = '../data'
    targets_file_path = os.path.join(folder_path, 'targets.csv')
    df_mpid = pd.read_csv(targets_file_path, header=None, names=['targets'])

    mpid_file_path = os.path.join(folder_path, 'mpid.csv')
    df_targets = pd.read_csv(mpid_file_path, header=None, names=['material_id'])

    dfA.reset_index(drop=True, inplace=True)
    df_mpid.reset_index(drop=True, inplace=True)
    df_targets.reset_index(drop=True, inplace=True)
    df_merged = pd.concat([dfA, df_mpid, df_targets], axis=1)

    # basedata_path = os.path.join(folder_path, 'element_data_20231115', 'all.csv')
    basedata_path = '../data/all.csv'
    df_basedata = pd.read_csv(basedata_path, index_col=0, low_memory=False)

    df_merged_A = pd.merge(df_merged, df_basedata, on='material_id')
    return df_merged_A


def prepare_data(df_merged_A):
    num_data_list = [
        'cluster',
        'cluster_sorted',
        'tSNE_1',
        'tSNE_2',
        'targets',
        'nsites',
        'nelements',
        'volume',
        'density',
        'density_atomic',
        'uncorrected_energy_per_atom',
        'energy_per_atom',
        'formation_energy_per_atom',
        'energy_above_hull',
        'equilibrium_reaction_energy_per_atom',
        'band_gap',
        'cbm',
        'vbm',
        'efermi',
        'total_magnetization',
        'total_magnetization_normalized_vol',
        'total_magnetization_normalized_formula_units',
        'num_magnetic_sites',
        'num_unique_magnetic_sites',
        'k_voigt',
        'k_reuss',
        'k_vrh',
        'g_voigt',
        'g_reuss',
        'g_vrh',
        'universal_anisotropy',
        'homogeneous_poisson',
        'e_total',
        'e_ionic',
        'e_electronic',
        'n',
        'e_ij_max',
        'weighted_surface_energy_EV_PER_ANG2',
        'weighted_surface_energy',
        'weighted_work_function',
        'surface_anisotropy',
        'shape_factor'
    ]
    df_merged_B = df_merged_A[num_data_list].copy()
    return df_merged_B


def sort_clusters(df_merged_B):
    df_sorted_by_targets = df_merged_B.groupby('cluster')['targets'].mean().sort_values(ascending=False)
    sorted_clusters = df_sorted_by_targets.index.tolist()
    cluster_mapping = {old_cluster: 11 - new_cluster for new_cluster, old_cluster in enumerate(sorted_clusters, start=1)}
    df_merged_B['cluster_sorted'] = df_merged_B['cluster'].replace(cluster_mapping)
    return df_merged_B, sorted_clusters


def plot_boxplots(df_merged_B, sorted_clusters, output_folder_path):
    # 数値列のみを選択し、プロットに不要な列を除外
    numeric_cols = df_merged_B.select_dtypes(include=['number']).columns.tolist()
    columns_to_plot = [col for col in numeric_cols if col not in ['cluster', 'cluster_sorted']]

    for col in columns_to_plot:
        try:
            plt.figure(figsize=(10, 6))
            sns.boxplot(
                x='cluster_sorted',
                y=col,
                data=df_merged_B,
                showfliers=False,
                # order=sorted_clusters,
                # hue='cluster_sorted',
                palette="rainbow"
            )
            plt.title(f'Boxplot of {col} by Cluster')
            file_name = f'{col}_boxplot.png'
            plt.savefig(os.path.join(output_folder_path, file_name))
            plt.close()
        except Exception as e:
            print(f"列 '{col}' のプロット中にエラーが発生しました: {e}")


def plot_scatter(df_merged_B, output_folder_path):

    fig = px.scatter(
        df_merged_B,
        x='tSNE_1',
        y='tSNE_2',
        height=600,
        width=700,
        color='total_magnetization',
        hover_name='formula_pretty',
        title="K-means Clustering Results",
        color_discrete_sequence=px.colors.qualitative.Vivid[::-1]  # カラースケールを逆にする
    )
    fig.show()

    fig = px.scatter(
        df_merged_B,
        x='tSNE_1',
        y='tSNE_2',
        height=600,
        width=700,
        color='cluster_sorted',
        hover_name='formula_pretty',
        title="K-means Clustering Results",
        color_discrete_sequence=px.colors.qualitative.Vivid[::-1]  # カラースケールを逆にする
    )
    fig.show()
    # グラフをHTMLファイルとして保存
    pio.write_html(fig, file=os.path.join(output_folder_path, 'kmeans_clustering_results.html'), auto_open=True)


def main(folder_path, project_path, target_data, model_name, n_clusters):

    output_folder_path = os.path.join(folder_path, project_path, f'{target_data}_{model_name}')
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # データの読み込み
    df = load_data(folder_path, model_name)

    # クラスタリングの実行
    df = perform_clustering(df, n_clusters=n_clusters)

    # クラスタのソート
    df, sorted_clusters = sort_clusters(df)

    # # 散布図の作成
    # plot_scatter(df_merged_A, output_folder_path)

    # # データの準備
    # df_merged_B = prepare_data(df_merged_A)

    # # ボックスプロットの作成
    # plot_boxplots(df_merged_B, sorted_clusters, output_folder_path)


if __name__ == "__main__":

    folder_path = '../data'
    model_name = 'MPNN'
    target_data = 'predicted_zT'
    project_path = f'{target_data}'
    n_clusters = 10

    main(folder_path, project_path, target_data, model_name, n_clusters)
