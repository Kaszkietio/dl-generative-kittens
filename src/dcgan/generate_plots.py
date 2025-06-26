import argparse
import os
import mlflow
import json
import matplotlib.pyplot as plt

def fetch_metrics_from_mlflow(run_id, metric_name):
    """
    Fetches the specified metric from an MLflow run.
    """
    client = mlflow.tracking.MlflowClient()
    metric_history = client.get_metric_history(run_id, metric_name)
    return [metric.value for metric in metric_history], [metric.step for metric in metric_history]

def plot_metrics(models, metrics, title, line_styles):
    """
    Plots the specified metrics for the given models.
    """
    for model_name, run_id in models:
        for metric_name, line_style in zip(metrics, line_styles):
            metric_name = "C_loss" if model_name.startswith("WGAN") and metric_name == "D_loss" else metric_name
            metric_values, steps = fetch_metrics_from_mlflow(run_id, metric_name)
            plt.plot(steps, metric_values, line_style, label=f"{model_name} - {metric_name}")

    plt.xlabel("Step")
    plt.ylabel("Metric Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

def main():
    parser = argparse.ArgumentParser(description="Generate plots for model metrics from MLflow.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration file (JSON format).")
    args = parser.parse_args()

    # Load configuration from the JSON file
    with open(args.config, "r") as config_file:
        config = json.load(config_file)


    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])

    for plot in config["plots"]:
        models = [(model["name"], model["run_id"]) for model in plot["models"]]

        plt.figure(figsize=(6, 4))

        plot_metrics(
            models,
            metrics=["G_loss", "D_loss"],
            title="Generator and Discriminator Loss",
            line_styles=["--", "-"]
        )

        os.makedirs(os.path.dirname(plot["output_path"]), exist_ok=True)
        plt.savefig(plot["output_path"])
        plt.close()

if __name__ == "__main__":
    main()