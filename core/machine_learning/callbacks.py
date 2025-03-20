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


class ModelCheckpoint:
    def __init__(self, maximize=False):
        self.maximize = maximize
        self.best_metric_value = float("-inf") if maximize else float("inf")
        self.best_model_weights = None
        
    def update_weights(self, model, metric_value):
        if self.outperforms_current(metric_value):
            self.best_metric_value = metric_value
            self.best_model_weights = model.state_dict()
            print(f"znaleziono nowy: {self.best_metric_value}")

    
    def outperforms_current(self, metric_value):
        if self.maximize:
            return self.best_metric_value < metric_value
        else:
            return self.best_metric_value > metric_value
        

    def get_best_weights(self):
        print(f"best_metric_value: {self.best_metric_value}")
        return self.best_model_weights