import os
import torch
import mlflow
import matplotlib.pyplot as plt

from core.utils.utilities import (
    generate_directory_path,
    ensure_directory_exists,
    save_config_params,
    log_params_saved,
    log_saved_file_path
)


def get_device_from_model(model):
    return next(model.parameters()).device


def transfer_to_device(features, targets, device):
    features = features.to(device)
    targets = targets.to(device)

    return features, targets


def log_training_start():
    print("Training has started...\n")


def log_epoch_results(epoch, train_metrics, valid_metrics):
    print(f"\nEpoch: {epoch + 1}")
    print(f"Training loss: {train_metrics[0]:.4f} | Validation loss: {valid_metrics[0]:.4f}")
    print(f"Training accuracy: {train_metrics[1]:.4f} | Validation accuracy: {valid_metrics[1]:.4f}\n")


def plot_training_curves(results: dict, metric: str) -> plt.Figure:
    num_epochs = len(results[metric]["train"])
    epoch_range = range(1, num_epochs + 1)
    
    fig, ax = plt.subplots()
    ax.plot(epoch_range, results[metric]["train"], label=f"Training {metric.capitalize()} Curve")
    ax.plot(epoch_range, results[metric]["valid"], label=f"Validation {metric.capitalize()} Curve")
    ax.set_title(f"{metric.capitalize()} Curves for Training and Validation Data")
    ax.set_xlabel("Consecutive Epochs")
    ax.set_ylabel(f"{metric.capitalize()}")
    ax.set_xlim((0, num_epochs + 1))
    ax.set_ylim((0, 100))
    ax.axis("equal")
    ax.grid(True)
    ax.legend()

    return fig



def prepare_directory(target_path: str, directory_name: str) -> str:
    directory_path = generate_directory_path(target_path, directory_name)
    ensure_directory_exists(directory_path) 

    return directory_path


def handle_model_saving(
        model,
        optimizer,
        lr_scheduler,
        figure1: plt.Figure,
        figure2: plt.Figure,
        target_path: str,
        params: dict = None        
) -> None:
    """
    opis
    """
    save_model(target_path, model)
    log_model_saved()

    checkpoints = create_checkpoints(optimizer, lr_scheduler)
    save_checkpoints(target_path, checkpoints)
    log_checkpoints_saved()

    save_plot(target_path, figure1, file_name="loss.png")
    save_plot(target_path, figure2, file_name="accuracy.png")
    log_plot_saved()

    if params:
        config_file_name = "model_parameters.txt"
        save_config_params(target_path, params, config_file_name)
        log_params_saved(config_file_name)

    log_saved_file_path(target_path)


def save_model(target_dir, model, file_name="model.pth"):
    torch.save(model.state_dict(), os.path.join(target_dir, file_name))


def log_model_saved():
    print("Model has been successfully saved.")

    
def create_checkpoints(optimizer, lr_scheduler) -> dict:
    return {
        "optimizer": optimizer.state_dict(),
        "scheduler": lr_scheduler,
    }


def save_checkpoints(target_dir, checkpoints, file_name="my_checkpoints.pth"):
    torch.save(checkpoints, os.path.join(target_dir, file_name))


def log_checkpoints_saved():
    print("Checkpoints have been successfully saved.")


def save_plot(target_dir, figure, file_name="plot.png"):
    figure.savefig(os.path.join(target_dir, file_name))


def log_plot_saved():
    print("Plot has been successfully saved.")


def initialize_results_tracker() -> dict:
    return {
        "loss": {
            "train": [],
            "valid": [],
        },
        "accuracy": {
            "train": [],
            "valid": [],
        },
    }


def setup_mlflow(parameters: dict) -> None:
    mlflow.set_tracking_uri(parameters["tracking_uri"])
    mlflow.set_experiment(parameters["experiment_name"])


def log_training_params(num_epochs, optimizer, lr_scheduler) -> None:
    mlflow.log_params({
        "num_epochs": num_epochs,
        "learning_rate": optimizer.param_groups[0]['lr'],
        "scheduler": str(lr_scheduler),
    })


def log_training_metrics(
    train_metrics: tuple, valid_metrics: tuple, num_epochs: int) -> None:
    mlflow.log_metrics(
        metrics={
            "train_loss": train_metrics[0],
            "valid_loss": valid_metrics[0],
            "train_accuracy": train_metrics[1],
            "valid_accuracy": valid_metrics[1],
        },
        step=num_epochs
    )


def log_model_artifacts(target_path: str) -> None:
    mlflow.log_artifacts(target_path)


def update_results_tracker(
    results: dict, train_metrics: tuple, valid_metrics: tuple) -> None:
    results["loss"]["train"].append(train_metrics[0])
    results["loss"]["valid"].append(valid_metrics[0])
    results["accuracy"]["train"].append(train_metrics[1])
    results["accuracy"]["valid"].append(valid_metrics[1])


def is_stopper_triggered(stopper, valid_loss: float) -> bool:
    return stopper and stopper.stop(valid_loss)


def log_early_stopping(stopper_name: str) -> None:
    print(f"Training stopped early due to '{stopper_name}' condition.")


def log_training_complete(model_name: str, total_epochs: int) -> None:
    print(f"Training of '{model_name}' completed after {total_epochs} epochs.\n")


def summarize_training(results: dict, evaluation: tuple) -> None:
    print("-- -- Training Summary: -- --")
    print(f"Number of Epochs: {len(results['loss']['train'])}\n")

    print(f"Final Training Loss: {results['loss']['train'][-1]:.4f}")
    print(f"Final Validation Loss: {results['loss']['valid'][-1]:.4f}\n")

    print(f"Final Training Accuracy: {results['accuracy']['train'][-1]:.4f}")
    print(f"Final Validation Accuracy: {results['accuracy']['valid'][-1]:.4f}\n")

    print("-- -- Model Evaluation: -- --")
    print(f"Test Loss: {evaluation[0]:.4f}")
    print(f"Test Accuracy: {evaluation[1]:.4f}\n")