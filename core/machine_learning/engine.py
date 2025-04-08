import torch
from tqdm.auto import tqdm

from .engine_setup import (
    initialize_model,
    initialize_training_components,
    initialize_callbacks
)

from .engine_utils import (
    get_device_from_model,
    transfer_data_to_device,
    log_training_start,
    log_epoch_results,
    initialize_results_tracker,
    log_training_params,
    log_training_metrics,
    update_results_tracker,
    is_stopper_triggered,
    log_early_stopping,
    log_training_complete,
    is_training_mode,
    get_context_manager,
    calculate_average,
    create_checkpoints
)


def setup_and_train_model(
        train_dataloader: torch.utils.data.DataLoader,
        valid_dataloader: torch.utils.data.DataLoader,
        model_parameters: dict
) -> torch.nn.Module:
    """
    opis
    """
    model = initialize_model(model_parameters)

    loss_fn, accuracy_fn, optimizer, lr_scheduler = initialize_training_components(
        model, model_parameters)

    init_stopper, early_stopper, model_checkpoint = initialize_callbacks(model_parameters)

    training_metrics = train_and_validate_model(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        init_stopper=init_stopper,
        early_stopper=early_stopper,
        model_checkpoint=model_checkpoint,
        num_epochs=model_parameters["num_epochs"])

    best_training_step = model_checkpoint.get_best_training_step()

    checkpoints = create_checkpoints(
        optimizer, lr_scheduler, training_metrics, best_training_step)

    return model, checkpoints
    

def train_and_validate_model(
        model, train_dataloader, valid_dataloader, loss_fn, accuracy_fn,
        optimizer, lr_scheduler, init_stopper=None, early_stopper=None, model_checkpoint=None,
        num_epochs=100,
):

    results_tracker = initialize_results_tracker()

    log_training_start()
    log_training_params(num_epochs, optimizer, lr_scheduler)

    for epoch in tqdm(range(1, num_epochs + 1)):
        train_metrics = perform_training_step(
            model, train_dataloader, loss_fn, accuracy_fn, optimizer, lr_scheduler)

        valid_metrics = perform_validation_step(
            model, valid_dataloader, loss_fn, accuracy_fn)

        log_epoch_results(epoch, train_metrics, valid_metrics)
        log_training_metrics(train_metrics, valid_metrics, epoch)
        update_results_tracker(results_tracker, train_metrics, valid_metrics)

        if is_stopper_triggered(init_stopper, valid_metrics[0]):
            log_early_stopping("init_stopper")
            break

        if is_stopper_triggered(early_stopper, valid_metrics[0]):
            log_early_stopping("early_stopper")
            break

        model_checkpoint.update_training_step(model, epoch, valid_metrics[0])

    log_training_complete("Seizure Detection Model", epoch)

    return results_tracker


def perform_training_step(model, dataloader, loss_fn, accuracy_fn, optimizer, lr_scheduler):
    loss, accuracy = perform_step(
        model, dataloader, loss_fn, accuracy_fn, optimizer, lr_scheduler)

    return loss, accuracy


def perform_validation_step(model, dataloader, loss_fn, accuracy_fn):
    loss, accuracy = perform_step(
        model, dataloader, loss_fn, accuracy_fn)

    return loss, accuracy


def perform_step(
        model,
        dataloader,
        loss_fn,
        accuracy_fn,
        optimizer=None,
        lr_scheduler=None
):
    training_mode = is_training_mode(optimizer)
    device = get_device_from_model(model)
    accumulated_accuracy = 0.0
    accumulated_loss = 0.0

    model.train() if training_mode else model.eval()

    with get_context_manager(training_mode):
        for features, targets in dataloader:
            features, targets = transfer_data_to_device(features, targets, device)
            predictions = model(features)

            accuracy = accuracy_fn(targets, predictions)    
            accumulated_accuracy += accuracy

            loss = loss_fn(predictions, targets)
            accumulated_loss += loss.item()

            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if training_mode and lr_scheduler:
            lr_scheduler.step()

    average_loss = calculate_average(accumulated_loss, dataloader)
    average_accuracy = calculate_average(accumulated_accuracy, dataloader)

    return average_loss, average_accuracy