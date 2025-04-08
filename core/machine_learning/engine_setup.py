from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from .model_builder import R2Plus1DConvNet
from .callbacks import InitStopper, EarlyStopper, ModelCheckpoint
from .accuracy import BinaryAccuracy


def initialize_model(model_parameters):
    model = R2Plus1DConvNet(
        in_channels=model_parameters["in_channels"],
        num_classes=model_parameters["num_classes"],
        dropout=model_parameters["dropout"]) 

    return model.to(model_parameters["device"])


def initialize_training_components(model, model_parameters):
    loss_fn, accuracy_fn = initialize_metrics()
    
    if model_parameters["optimizer"] == "SGD":
        optimizer = SGD(
            params=model.parameters(),
            lr=model_parameters["learning_rate"],
            momentum=model_parameters["momentum"],
            weight_decay=model_parameters["weight_decay"]
        )
    elif model_parameters["optimizer"] == "Adam":
        optimizer = Adam(
            params=model.parameters(),
            lr=model_parameters["learning_rate"],
            weight_decay=model_parameters["weight_decay"]
        )
    
    if model_parameters["lr_scheduler"] == "step":
        lr_scheduler = StepLR(
            optimizer,
            step_size=model_parameters["step_size"],
            gamma=model_parameters["gamma"]
        )
    elif model_parameters["lr_scheduler"] == "cosine":
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=model_parameters["t_max"],
            eta_min=model_parameters["eta_min"]
        )     
    
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
    
    model_checkpoint = ModelCheckpoint(
        maximize=model_parameters["maximize"])
    
    return init_stopper, early_stopper, model_checkpoint