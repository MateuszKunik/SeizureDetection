import sys

if sys.platform == "win32":
    sys.path.append("D:/DevSpace/Projects/Research/SeizureDetection")
elif sys.platform == "linux":
    sys.path.append("/mnt/d/gniazdko/SeizureDetection")

import mlflow
from core.utils import ProjectManager, ConfigManager
from core.utils import load_model_input_data

from core.machine_learning import (
    setup_mlflow,
    sample_data,
    split_data_by_proportions,
    create_dataloaders,
    setup_and_train_model,
    plot_training_curves,
    summarize_training,
    handle_model_saving,
    evaluate_model_performance,
    prepare_directory,
    log_model_artifacts
)



project_manager = ProjectManager()
configs_directory_path = project_manager.get_configs_directory_path()
primary_data_path = project_manager.get_primary_data_path()
model_directory_path = project_manager.get_model_data_path()

config_manager = ConfigManager(configs_directory_path)
model_params = config_manager.load_config("parameters_machine_learning")


# model_input_data = load_model_input_data(
#     primary_data_path, model_params["data_version"])

###########
import numpy as np

images, labels = load_model_input_data(primary_data_path, model_params["data_version"])
images = images.reshape(-1, 64, 64, 1)
labels = np.repeat(labels, 18)
model_input_data = images, labels
###########

model_input_data = sample_data(model_input_data, model_params["data_parameters"]["data_fraction"])


setup_mlflow(model_params["mlflow_parameters"])

with mlflow.start_run():
    data_splits = split_data_by_proportions(
        model_input_data, model_params["data_parameters"])

    train_data, valid_data, test_data = create_dataloaders(
        data_splits, model_params["data_parameters"])
    # ####
    # dataloader = next(iter(train_data))
    # print(type(dataloader))
    # print(dataloader[0].shape)
    # print(dataloader[0][0].dtype)
    # print(dataloader[1].shape)
    # print(f"example: {dataloader[1][0]}")
    # print(f"type: {dataloader[1][0].dtype}")
    # ####
    model, optimizer, lr_scheduler, results = setup_and_train_model(
        train_data, valid_data, model_params["model_parameters"])

    evaluation_metrics = evaluate_model_performance(model, test_data)

    directory_path = prepare_directory(
        model_directory_path, model_params["model_name"])

    log_model_artifacts(directory_path)
    summarize_training(results, evaluation_metrics)
    loss_figure = plot_training_curves(results, metric="loss")
    accuracy_figure = plot_training_curves(results, metric="accuracy")

    handle_model_saving(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        figure1=loss_figure,
        figure2=accuracy_figure,
        target_path=directory_path,
        params=model_params)