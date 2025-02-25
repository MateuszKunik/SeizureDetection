import torch
import torch.nn as nn


class BinaryAccuracy(nn.Module):
    """
    Computes the accuracy for binary classification.
    """
    def __init__(self):
        super().__init__()
    

    def forward(self, targets, logits):
        actual_labels = self.get_actual_labels(targets)
        predicted_labels = self.compute_predicted_labels(logits)
        
        return self.calculate_accuracy(actual_labels, predicted_labels)
    

    def get_actual_labels(self, targets):
        if len(targets.shape) > 1:
            actual_labels = torch.argmax(targets, dim=1)
        else:
            actual_labels = targets.int()
        
        return actual_labels
    

    def compute_predicted_labels(self, logits):
        label_probabilities = torch.softmax(logits, dim=1)
        predicted_labels = torch.argmax(label_probabilities, dim=1) 
        
        return predicted_labels
    

    def calculate_accuracy(self, actual_labels, predicted_labels):
        correct_predictions = (actual_labels == predicted_labels).sum().item()
        accuracy = correct_predictions / actual_labels.size(0)
        
        return accuracy