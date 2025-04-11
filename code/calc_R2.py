import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from urllib.error import HTTPError
import plotly.express as px
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os


def main(file_path, data_name):
    # # Load the data from the specified CSV file
    # file_path = f'/Users/yusukehashimoto/Library/CloudStorage/GoogleDrive-yusuke.hashimoto.b8@tohoku.ac.jp/マイドライブ/MatDeepLearn/output/A241126/counts_{pre_fc_count}_{gc_count}_{post_fc_count}/bulk_data_bulk_data_C/bulk_data_bulk_data_C_{model}_demo_50_DR_output.csv'

    # Add retry mechanism to handle too many requests
    data = pd.read_csv(file_path, names=['index', 'target', 'PCA_1', 'PCA_2', 'tSNE_1', 'tSNE_2'], skiprows=1)

    # Extract tSNE_1 and tSNE_2 columns as coordinates
    coordinates = data[['tSNE_1', 'tSNE_2']].values

    # Calculate the pairwise distance between all points
    distances = np.linalg.norm(coordinates[:, np.newaxis] - coordinates[np.newaxis, :], axis=-1)

    # Set diagonal (self-distance) to infinity to ignore them in the calculation
    np.fill_diagonal(distances, np.inf)  # Correctly ignore self-distance

    # Find the index of the nearest point for each point
    nearest_indices = distances.argmin(axis=1)

    # Calculate the difference in target values between each point and its nearest point
    target_values = data['target'].values
    nearest_target_values = target_values[nearest_indices]
    target_differences = target_values - nearest_target_values

    # Create a DataFrame for easier visualization
    nearest_distances_corrected_df = pd.DataFrame({
        'Point Index': data.index,
        'Nearest Distance': distances.min(axis=1),
        'Nearest Point Index': nearest_indices,
        'Target Difference': target_differences,
        'Target Value': target_values,
        'Nearest Target Value': nearest_target_values
    })

    # Calculate and print evaluation metrics
    mse = mean_squared_error(target_values, nearest_target_values)
    mae = mean_absolute_error(target_values, nearest_target_values)
    r2 = r2_score(target_values, nearest_target_values)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R^2 Score: {r2}")

    # Store evaluation metrics in a dictionary
    metrics = {
        'Model': data_name,
        # 'GC Count': gc_count,
        'MSE': mse,
        'MAE': mae,
        'R2': r2
    }

    # Calculate and print the root mean square of the nearest distances
    mean_nearest_distance = np.sqrt((nearest_distances_corrected_df['Nearest Distance'] ** 2).mean())
    std_nearest_distance = np.sqrt((nearest_distances_corrected_df['Nearest Distance'] ** 2).std())
    print(f"RMS of Nearest Distances: {mean_nearest_distance}")
    print(f"RMS Standard Deviation of Nearest Distances: {std_nearest_distance}")

    # Calculate and print the root mean square of the target differences
    mean_target_difference = np.sqrt((nearest_distances_corrected_df['Target Difference'] ** 2).mean())
    std_target_difference = np.sqrt((nearest_distances_corrected_df['Target Difference'] ** 2).std())
    print(f"RMS of Target Differences: {mean_target_difference}")
    print(f"RMS Standard Deviation of Target Differences: {std_target_difference}")

    # Plot histogram of the nearest distances
    plt.figure(figsize=(10, 6))
    plt.hist(nearest_distances_corrected_df['Nearest Distance'], bins=30, range=(0, 3), edgecolor='black', alpha=0.7)
    plt.xlabel('Nearest Distance')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Nearest Distances for {data_name}')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f'histogram_nearest_distances_{data_name}.png')  # Save the histogram with model name
    plt.close()

    return mean_nearest_distance, std_nearest_distance, metrics


def plot_scatter(df_r):
    # Melt the DataFrame for easier plotting with color distinction by model
    df_melted = df_r.reset_index().melt(id_vars='index', var_name='Model', value_name='Mean Nearest Distance')

    # Plot a scatter plot of mean nearest distances for different models and GC counts
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_melted, x='index', y='Mean Nearest Distance', hue='Model', palette='viridis', s=100)
    plt.title('Scatter Plot of Mean Nearest Distances for Different Models and GC Counts')
    plt.xlabel('GC Count')
    plt.ylabel('Mean Nearest Distance')
    plt.legend(title='Model')
    plt.grid(True)
    plt.ylim(0, 1)  # Set y-axis range from 0 to 1
    plt.show()


def plot_line(df_r):
    # Plot a line plot of the DataFrame using Plotly with y-axis limited to 0 to 1
    fig = px.line(df_r.T)
    fig.update_yaxes(range=[0, 1])  # Set y-axis range from 0 to 1
    fig.show()


if __name__ == '__main__':
    pre_fc_count = 1
    post_fc_count = 3
    list_model = ['CGCNN', 'MPNN', 'GCN', 'SchNet', 'MEGNet']
    mean_distances = []
    metrics_list = []

    df_r = pd.DataFrame()

    # for gc_count in range(1, 10, 1):
    gc_count = 10
    gc_mean_distances = []

    list_data_name = ['no_GRU', 'no_GRU_NN', 'no_NN', 'original']
    for data_name in list_data_name:
        # Load the data from the specified CSV file
        folder_path = '../data/GRU_NN'
        file_path = os.path.join(folder_path, f'{data_name}_counts_{pre_fc_count}_{gc_count}_{post_fc_count}.csv')

        mean_nearest_distance, std_nearest_distance, metrics = main(file_path, data_name)
        print(mean_nearest_distance)
        df_r.loc[gc_count, data_name] = mean_nearest_distance
        metrics_list.append(metrics)

    print(df_r)

    # Convert metrics to DataFrame and display
    metrics_df = pd.DataFrame(metrics_list)
    print(metrics_df)
