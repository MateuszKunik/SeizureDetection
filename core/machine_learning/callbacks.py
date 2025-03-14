class InitStopper:
    def __init__(self, patience=1):
        self.patience = patience
        self.init_validation_loss = None
        self.counter = 0

    def stop(self, validation_loss):
        if self.init_validation_loss:
            if self.init_validation_loss == validation_loss:
                self.counter += 1
                if self.counter >= self.patience:
                    return True         
        else:
            self.init_validation_loss = validation_loss

        return False


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# import torch

# from .engine_utils import save_model

# class ModelCheckpoint:
#     def __init__(self, model, filepath="best_model.pth", mode="min"):
#         self.model = model
#         self.filepath = filepath
#         self.mode = mode
#         self.best_value = float("inf") if mode == "min" else float("-inf")
    
#     def save(self, validation_loss):
#         if (self.mode == "min" and validation_loss < self.best_value) or \
#            (self.mode == "max" and validation_loss > self.best_value):
#             self.best_value = validation_loss
            
