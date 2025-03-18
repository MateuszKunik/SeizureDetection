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
    log_model_artifacts,
    print_classification_report
)

wiecej_danych = False

project_manager = ProjectManager()
configs_directory_path = project_manager.get_configs_directory_path()
primary_data_path = project_manager.get_primary_data_path()
model_directory_path = project_manager.get_model_data_path()

config_manager = ConfigManager(configs_directory_path)
model_params = config_manager.load_config("parameters_machine_learning")


model_input_data = load_model_input_data(primary_data_path, model_params)

model_input_data = sample_data(
    model_input_data, model_params["data_parameters"]["data_fraction"])


setup_mlflow(model_params["mlflow_parameters"])

with mlflow.start_run():
    data_splits = split_data_by_proportions(
        model_input_data, model_params["data_parameters"])

    # plac budowy
    if wiecej_danych:
        import numpy as np

        X_train, y_train = data_splits["train"]
        # print(X_train.shape), print(y_train.shape)

        rng = np.random.default_rng(1024)
        X_train_perm = rng.permutation(X_train, axis = 2)

        X_train = np.concatenate([X_train, X_train_perm])
        y_train = np.concatenate([y_train, y_train])

        # print(X_train.shape), print(y_train.shape)

        data_splits["train"] = X_train, y_train
    #####

    train_data, valid_data, test_data = create_dataloaders(
        data_splits, model_params["data_parameters"])

    model, optimizer, lr_scheduler, results = setup_and_train_model(
        train_data, valid_data, model_params["model_parameters"])

    directory_path = prepare_directory(
        model_directory_path, model_params["model_name"])

    log_model_artifacts(directory_path)
    summarize_training(results)

    classification_report = evaluate_model_performance(model, test_data)
    print_classification_report(classification_report)

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
    

