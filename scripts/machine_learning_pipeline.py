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
    handle_training_artifacts_saving,
    generate_evaluation_report,
)


project_manager = ProjectManager()
configs_directory_path = project_manager.get_configs_directory_path()
primary_data_path = project_manager.get_primary_data_path()
model_directory_path = project_manager.get_model_data_path()

config_manager = ConfigManager(configs_directory_path)
model_parameters = config_manager.load_config("parameters_machine_learning")


model_input_data = load_model_input_data(primary_data_path, model_parameters)

model_input_data = sample_data(
    model_input_data, model_parameters["data_parameters"]["data_fraction"])


setup_mlflow(model_parameters["mlflow_parameters"])

with mlflow.start_run():
    data_splits = split_data_by_proportions(
        model_input_data, model_parameters["data_parameters"])

    train_dataloader, valid_dataloader, test_dataloader = create_dataloaders(
        data_splits, model_parameters["data_parameters"])

    classification_model, checkpoints = setup_and_train_model(
        train_dataloader,
        valid_dataloader,
        model_parameters["model_parameters"])

    evaluation_report = generate_evaluation_report(
        classification_model,
        test_dataloader,
        checkpoints["best_training_step"],
        model_parameters["model_parameters"])

    handle_training_artifacts_saving(
        classification_model,
        checkpoints,
        model_parameters,
        evaluation_report,
        model_directory_path)