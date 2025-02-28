import torch
from tqdm.auto import tqdm
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from .model_builder import SimpleNet, STConvNet, TinyVGG
from .accuracy import BinaryAccuracy
from .callbacks import InitStopper, EarlyStopper

from .engine_utils import (
    get_device_from_model,
    transfer_to_device,
    log_training_start,
    log_epoch_results,
    initialize_results_tracker,
    log_training_params,
    log_training_metrics,
    update_results_tracker,
    is_stopper_triggered,
    log_early_stopping,
    log_training_complete,
)


def setup_and_train_model(
        train_dataloader: torch.utils.data.DataLoader,
        valid_dataloader: torch.utils.data.DataLoader,
        model_parameters: dict
) -> torch.nn.Module:
    """
    opis
    """
    classification_model = initialize_model(model_parameters)

    loss_fn, accuracy_fn, optimizer, lr_scheduler = initialize_training_components(
        classification_model, model_parameters)

    init_stopper, early_stopper = initialize_callbacks(model_parameters)

    results = train_and_validate_model(
        model=classification_model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        init_stopper=init_stopper,
        early_stopper=early_stopper,
        num_epochs=model_parameters["num_epochs"])

    return classification_model, optimizer, lr_scheduler, results
    

def initialize_model(model_parameters):
    model = TinyVGG(
        in_channels=model_parameters["in_channels"],
        num_classes=model_parameters["num_classes"],
        dropout=model_parameters["dropout"]) 

    # model = STConvNet(
    #     in_channels=model_parameters["in_channels"],
    #     num_classes=model_parameters["num_classes"],
    #     dropout=model_parameters["dropout"])

    return model.to(model_parameters["device"])


def initialize_training_components(model, model_parameters):
    loss_fn, accuracy_fn = initialize_metrics()
    
    if model_parameters["optimizer"] == "SGD":
        optimizer = SGD(
            params=model.parameters(),
            lr=model_parameters["learning_rate"],
            momentum=model_parameters["momentum"],
            weight_decay=model_parameters["weight_decay"])
        
    elif model_parameters["optimizer"] == "Adam":
        optimizer = Adam(
            params=model.parameters(),
            lr=model_parameters["learning_rate"],
            weight_decay=model_parameters["weight_decay"])
    
    if model_parameters["lr_scheduler"] == "step":
        lr_scheduler = StepLR(
            optimizer,
            step_size=model_parameters["step_size"],
            gamma=model_parameters["gamma"])
        
    elif model_parameters["lr_scheduler"] == "cosine":
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=model_parameters["t_max"],
            eta_min=model_parameters["eta_min"])     
    
    return loss_fn, accuracy_fn, optimizer, lr_scheduler


def initialize_metrics():
    loss_fn = CrossEntropyLoss()
    accuracy_fn = BinaryAccuracy()

    return loss_fn, accuracy_fn
    

def initialize_callbacks(model_parameters):
    init_stopper = InitStopper(
        patience=model_parameters["init_stopper_patience"])
    
    early_stopper = EarlyStopper(
        patience=model_parameters["early_stopper_patience"],
        min_delta=model_parameters["early_stopper_min_delta"])
    
    return init_stopper, early_stopper


def train_and_validate_model(
        model, train_dataloader, valid_dataloader, loss_fn, accuracy_fn,
        optimizer, lr_scheduler, init_stopper=None, early_stopper=None,
        num_epochs=100,
):

    results_tracker = initialize_results_tracker()

    log_training_start()
    log_training_params(num_epochs, optimizer, lr_scheduler)

    for epoch in tqdm(range(num_epochs)):
        train_metrics = perform_training_step(
            model, train_dataloader, loss_fn, accuracy_fn, optimizer, lr_scheduler)

        valid_metrics = perform_validation_step(model, valid_dataloader, loss_fn, accuracy_fn)

        log_epoch_results(epoch, train_metrics, valid_metrics)
        log_training_metrics(train_metrics, valid_metrics, epoch)
        update_results_tracker(
            results_tracker, train_metrics, valid_metrics)

        if is_stopper_triggered(init_stopper, valid_metrics[0]):
            log_early_stopping("init_stopper")
            break

        if is_stopper_triggered(early_stopper, valid_metrics[0]):
            log_early_stopping("early_stopper")
            break

    log_training_complete("regression_model", epoch+1)

    return results_tracker


def perform_training_step(model, dataloader, loss_fn, accuracy_fn, optimizer, lr_scheduler):
    loss, accuracy = perform_step(
        model, dataloader, loss_fn, accuracy_fn, optimizer, lr_scheduler)

    return loss, accuracy


def perform_validation_step(model, dataloader, loss_fn, accuracy_fn):
    loss, accuracy = perform_step(
        model, dataloader, loss_fn, accuracy_fn)

    return loss, accuracy


def evaluate_model_performance(model, dataloader):
    loss_fn, accuracy_fn = initialize_metrics()
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
            features, targets = transfer_to_device(features, targets, device)
            # print(features.dtype) #torch.float32
            predictions = model(features)
            # print("Engine")
            # print(features.shape, targets.shape, predictions.shape)
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


def is_training_mode(optimizer) -> bool:
    return bool(optimizer)


def get_context_manager(training_mode):
    return torch.set_grad_enabled(True) if training_mode else torch.inference_mode()


def calculate_average(metric, dataloader):
        return metric / len(dataloader)