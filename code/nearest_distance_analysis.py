import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from urllib.error import HTTPError
import plotly.express as px


def create_output_directory():
    output_dir = "nearest_distance_analysis_output"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def main(model, pre_fc_count, gc_count, post_fc_count, output_dir):
    # Load the data from the specified CSV file
    file_path = f'../data/model/bulk_data_bulk_data_C_{model}_demo_50_DR_output.csv'

    # Add retry mechanism to handle too many requests
    data = pd.read_csv(file_path, names=['index', 'target', 'PCA_1', 'PCA_2', 'tSNE_1', 'tSNE_2'], skiprows=1)

    # Extract tSNE_1 and tSNE_2 columns as coordinates
    coordinates = data[['tSNE_1', 'tSNE_2']].values

    # Calculate the pairwise distance between all points
    distances = np.linalg.norm(coordinates[:, np.newaxis] - coordinates[np.newaxis, :], axis=-1)

    # Set diagonal (self-distance) to infinity to ignore them in the calculation
    np.fill_diagonal(distances, np.inf)  # Ignore self-distance to focus on inter-point distances

    # Find the index of the nearest point for each point
    nearest_indices = distances.argmin(axis=1)

    # Calculate the difference in target values between each point and its nearest point
    target_differences = data['target'].values - data['target'].values[nearest_indices]

    # Create a DataFrame for easier visualization
    nearest_distances_corrected_df = pd.DataFrame({
        'Point Index': data.index,
        'Nearest Distance': distances.min(axis=1),
        'Nearest Point Index': nearest_indices,
        'Target Difference': target_differences
    })

    # Save the DataFrame to a CSV file
    output_file = os.path.join(output_dir, f'nearest_distances_{model}_gc_{gc_count}.csv')
    nearest_distances_corrected_df.to_csv(output_file, index=False)

    # Calculate and print the mean, standard deviation, and median of the nearest distances
    mean_nearest_distance = np.sqrt((nearest_distances_corrected_df['Nearest Distance'] ** 2).mean())
    std_nearest_distance = np.std(nearest_distances_corrected_df['Nearest Distance'])
    median_nearest_distance = np.median(nearest_distances_corrected_df['Nearest Distance'])
    print(f"RMS of Nearest Distances: {mean_nearest_distance}")
    print(f"Standard Deviation of Nearest Distances: {std_nearest_distance}")
    print(f"Median of Nearest Distances: {median_nearest_distance}")

    # Calculate and print additional metrics for target differences
    mean_target_difference = np.sqrt((nearest_distances_corrected_df['Target Difference'] ** 2).mean())
    std_target_difference = np.std(nearest_distances_corrected_df['Target Difference'])
    median_target_difference = np.median(np.abs(nearest_distances_corrected_df['Target Difference']))
    max_target_difference = np.max(np.abs(nearest_distances_corrected_df['Target Difference']))
    print(f"RMS of Target Differences: {mean_target_difference}")
    print(f"Standard Deviation of Target Differences: {std_target_difference}")
    print(f"Median of Target Differences: {median_target_difference}")
    print(f"Max of Target Differences: {max_target_difference}")

    # Plot histogram of the nearest distances
    plt.figure(figsize=(10, 6))
    plt.hist(nearest_distances_corrected_df['Nearest Distance'], bins=30, range=(0, 3), edgecolor='black', alpha=0.7)
    plt.xlabel('Nearest Distance')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Nearest Distances for {model} (GC Count: {gc_count})')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, f'histogram_nearest_distances_{model}_gc_{gc_count}.png'))  # Save the histogram
    plt.close()

    return mean_nearest_distance, std_nearest_distance, median_nearest_distance, mean_target_difference, std_target_difference, median_target_difference, max_target_difference


def plot_metrics(df_metrics, output_dir):
    # Create a line plot for each metric across models and GC counts
    metrics = ['Mean Nearest Distance', 'Standard Deviation of Nearest Distance', 'Median Nearest Distance',
               'RMS of Target Differences', 'Standard Deviation of Target Differences', 'Median of Target Differences', 'Max of Target Differences']

    for metric in metrics:
        plt.figure(figsize=(12, 8))
        for model in df_metrics.columns.levels[0]:
            plt.plot(df_metrics.index, df_metrics[model, metric], label=f'{model} - {metric}')
        plt.title(f'{metric} Across GC Counts')
        plt.xlabel('GC Count')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{metric.replace(" ", "_")}_plot.png'))  # Save the plot
        plt.close()


if __name__ == '__main__':
    output_dir = '../result'
    pre_fc_count = 1
    gc_count = 4
    post_fc_count = 3
    list_model = ['CGCNN', 'MPNN', 'GCN', 'SchNet', 'MEGNet']
    metrics = ['Mean Nearest Distance', 'Standard Deviation of Nearest Distance', 'Median Nearest Distance',
               'RMS of Target Differences', 'Standard Deviation of Target Differences', 'Median of Target Differences', 'Max of Target Differences']

    df_metrics = pd.DataFrame(columns=pd.MultiIndex.from_product([list_model, metrics]))

    for model in list_model:
        print(model, gc_count)
        results = main(model, pre_fc_count, gc_count, post_fc_count, output_dir)
        df_metrics.loc[gc_count, model] = results

    print(df_metrics)

    # Save the metrics DataFrame to a CSV file
    metrics_output_file = os.path.join(output_dir, 'metrics_summary.csv')
    df_metrics.to_csv(metrics_output_file)

    # Plot metrics
    plot_metrics(df_metrics, output_dir)
