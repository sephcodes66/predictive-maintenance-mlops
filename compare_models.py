import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
EXPERIMENT_NAME = "Predictive Maintenance"
OUTPUT_DIR = "visualizations"
OUTPUT_FILE = "performance_comparison.png"

# --- MLflow Client ---
client = mlflow.tracking.MlflowClient()

# --- Get Experiment ---
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if not experiment:
    print(f"Experiment '{EXPERIMENT_NAME}' not found.")
    exit()

# --- Find Latest Runs ---
runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])

baseline_run = None
tuning_run = None

for run in runs:
    if run.data.tags.get('mlflow.runName') == 'Tuning Run' and not tuning_run:
        tuning_run = run
    elif run.data.tags.get('mlflow.runName') != 'Tuning Run' and not baseline_run:
        baseline_run = run
    if baseline_run and tuning_run:
        break

if not baseline_run or not tuning_run:
    print("Could not retrieve one or both model runs. Exiting.")
    exit()

# --- Extract Metrics ---
baseline_metrics = baseline_run.data.metrics
tuned_metrics = tuning_run.data.metrics

# --- Prepare Data for Plotting ---
metrics_to_plot = ['mae', 'mse', 'r2']
data = {'Model': [], 'Metric': [], 'Value': []}

for metric in metrics_to_plot:
    if metric in baseline_metrics:
        data['Model'].append('Baseline')
        data['Metric'].append(metric.upper())
        data['Value'].append(baseline_metrics[metric])
    if metric in tuned_metrics:
        data['Model'].append('Tuned')
        data['Metric'].append(metric.upper())
        data['Value'].append(tuned_metrics[metric])

df = pd.DataFrame(data)

# --- Create and Save Plot ---
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
barplot = sns.barplot(x='Metric', y='Value', hue='Model', data=df, palette="viridis")

# Add labels to the bars
for p in barplot.patches:
    barplot.annotate(format(p.get_height(), '.4f'),
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'center',
                   xytext = (0, 9),
                   textcoords = 'offset points')

plt.title('Baseline vs. Tuned Model Performance', fontsize=16)
plt.ylabel('Metric Value', fontsize=12)
plt.xlabel('Metric', fontsize=12)
plt.legend(title='Model')
plt.tight_layout()

# --- Save the plot ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
plot_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
plt.savefig(plot_path)

print(f"Comparison plot saved to {plot_path}")