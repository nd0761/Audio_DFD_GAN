import os
import json
import pandas as pd

def load_json_files(directory):
    json_data = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r') as f:
                data = json.load(f)
                json_data.append(data)
    return json_data

def compute_averages(json_data):
    aggregate_sums = {
        "on noised": {},
        "on clean": {},
        "on all": {}
    }
    
    for data in json_data:
        for condition, metrics in data.items():
            if condition not in aggregate_sums:
                aggregate_sums[condition] = {}
            for metric, value in metrics.items():
                if metric not in aggregate_sums[condition]:
                    aggregate_sums[condition][metric] = 0
                aggregate_sums[condition][metric] += value
    
    average_metrics = {}
    for condition, metrics in aggregate_sums.items():
        average_metrics[condition] = {m:v / len(json_data) for m, v in metrics.items()}
    
    return average_metrics

def display_averages(average_metrics):
    df = pd.DataFrame.from_dict(average_metrics, orient='index')
    # import ace_tools as tools; tools.display_dataframe_to_user(name="Average Metrics Table", dataframe=df)
    return df

if __name__ == "__main__":
    logs_directory = '/tank/local/ndf3868/GODDS/GAN/logs/060824-10:07:07'  # Replace with your directory path

    # Load JSON files and compute averages
    json_data = load_json_files(logs_directory)
    average_metrics = compute_averages(json_data)

    # Display the averages
    average_metrics_df = display_averages(average_metrics)
    average_metrics_df.to_csv(os.path.join(logs_directory, 'average_logs.csv'))
