import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import json
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, top_k_accuracy_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

# This class is used to train a model using PyTorch and evaluate its performance.
class Trainer:
    # The constructor initializes the Trainer class with the model, learning rate, weight decay, batch size, number of epochs, and class weights.
    def __init__(self, model, learning_rate=0.0001, weight_decay=0.0001, batch_size=32, epochs=10, class_weights=None):
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # This trains the model using the provided training data and evaluates its performance on validation and test data if provided. 
    # For style classification, both validation and test data are provided. For architect, only test data is provided due to the limited number of samples.
    def train(self, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None, history_path="training_history.json", extra_info=None, label_encoder=None):
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long, device=self.device)
        train_data = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        best_epoch = None

        if X_val is not None and y_val is not None:
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=self.device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long, device=self.device)
        else:
            X_val_tensor, y_val_tensor = None, None

        if X_test is not None and y_test is not None:
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=self.device)
            y_test_tensor = torch.tensor(y_test, dtype=torch.long, device=self.device)
        else:
            X_test_tensor, y_test_tensor = None, None
        
        train_accuracies_epoch = []
        val_accuracies_epoch = []
        test_accuracies_epoch = []
        
        # train the model for the specified number of epochs.
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            self.model.train()
            with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}", ncols=100) as pbar:
                for X_batch, y_batch in train_loader:
                    # self.model.train()
                    self.optimizer.zero_grad()
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)
                    loss.backward()
                    self.optimizer.step()
                    pbar.update(1)
            
            # Evaluate training accuracy for the epoch.
            self.model.eval()
            with torch.no_grad():
                train_outputs = self.model(X_train_tensor)
                train_pred = torch.argmax(train_outputs, dim=1)
                epoch_train_acc = accuracy_score(y_train_tensor.cpu().numpy(), train_pred.cpu().numpy())
                train_accuracies_epoch.append(epoch_train_acc)
                
                # Evaluate validation accuracy if provided.
                if X_val_tensor is not None and y_val_tensor is not None:
                    val_outputs = self.model(X_val_tensor)
                    val_pred = torch.argmax(val_outputs, dim=1)
                    epoch_val_acc = accuracy_score(y_val_tensor.cpu().numpy(), val_pred.cpu().numpy())
                    val_accuracies_epoch.append(epoch_val_acc)
                    print(f"Epoch {epoch+1}: Train Acc: {epoch_train_acc:.4f} | Val Acc: {epoch_val_acc:.4f}")

                 # Evaluate test accuracy for each epoch if validation not provided.
                elif X_val_tensor is None and y_val_tensor is None:
                    test_outputs = self.model(X_test_tensor)
                    test_pred = torch.argmax(test_outputs, dim=1)
                    epoch_test_acc = accuracy_score(y_test_tensor.cpu().numpy(), test_pred.cpu().numpy())
                    test_accuracies_epoch.append(epoch_test_acc)
                    print(f"Epoch {epoch+1}: Train Acc: {epoch_train_acc:.4f} | Test Acc: {epoch_test_acc:.4f}")
                else:
                    print(f"Epoch {epoch+1}: Train Acc: {epoch_train_acc:.4f}")

        # Final evaluations.
        self.model.eval()
        with torch.no_grad():
            # training accuracy, including top-1 accuracy, top3-accuracy, precision, recall
            train_outputs = self.model(X_train_tensor)
            train_pred = torch.argmax(train_outputs, dim=1)
            final_train_acc = accuracy_score(y_train_tensor.cpu().numpy(), train_pred.cpu().numpy())
            train_precision = precision_score(y_train_tensor.cpu(), train_pred.cpu(), average=None, zero_division=0)
            train_recall = recall_score(y_train_tensor.cpu(), train_pred.cpu(), average=None, zero_division=0)
            top3_train_acc = Trainer.compute_top_k_accuracy(train_outputs, y_train_tensor, k=3)
            unique_labels = np.unique(y_train_tensor.cpu().numpy())
            if label_encoder is not None:
                label_names = label_encoder.inverse_transform(unique_labels)
            else:
                label_names = [str(i) for i in unique_labels]

            train_precision_dict = {label: float(score) for label, score in zip(label_names, train_precision)}
            train_recall_dict = {label: float(score) for label, score in zip(label_names, train_recall)}

            # Final validation evaluation metrics.
            final_val_acc = None
            if X_val_tensor is not None and y_val_tensor is not None:
                val_outputs = self.model(X_val_tensor)
                val_pred = torch.argmax(val_outputs, dim=1)
                final_val_acc = accuracy_score(y_val_tensor.cpu().numpy(), val_pred.cpu().numpy())
                val_precision = precision_score(y_val_tensor.cpu(), val_pred.cpu(), average=None, zero_division=0)
                val_recall = recall_score(y_val_tensor.cpu(), val_pred.cpu(), average=None, zero_division=0)
                top3_val_acc = Trainer.compute_top_k_accuracy(val_outputs, y_val_tensor, k=3)
                val_precision_dict = {label: float(score) for label, score in zip(label_names, val_precision)}
                val_recall_dict = {label: float(score) for label, score in zip(label_names, val_recall)}
            
            # Final test evaluation metrics.
            final_test_acc = None
            if X_test_tensor is not None and y_test_tensor is not None:
                test_outputs = self.model(X_test_tensor)
                test_pred = torch.argmax(test_outputs, dim=1)
                final_test_acc = accuracy_score(y_test_tensor.cpu().numpy(), test_pred.cpu().numpy())
                test_precision = precision_score(y_test_tensor.cpu(), test_pred.cpu(), average=None, zero_division=0)
                test_recall = recall_score(y_test_tensor.cpu(), test_pred.cpu(), average=None, zero_division=0)
                top3_test_acc = Trainer.compute_top_k_accuracy(test_outputs, y_test_tensor, k=3)
                test_precision_dict = {label: float(score) for label, score in zip(label_names, test_precision)}
                test_recall_dict = {label: float(score) for label, score in zip(label_names, test_recall)}

            # Confusion matrix for training, validation, and test sets.
            train_conf_matrix = confusion_matrix(y_train_tensor.cpu(), train_pred.cpu())
            if X_val_tensor is not None and y_val_tensor is not None:
                val_conf_matrix = confusion_matrix(y_val_tensor.cpu(), val_pred.cpu())

            if X_test_tensor is not None and y_test_tensor is not None:
                test_conf_matrix = confusion_matrix(y_test_tensor.cpu(), test_pred.cpu())

        print(f"Final Training Accuracy: {final_train_acc:.4f}")
        if final_val_acc is not None:
            print(f"Final Validation Accuracy: {final_val_acc:.4f}")
        if final_test_acc is not None:
            print(f"Final Test Accuracy: {final_test_acc:.4f}")
        print(f"Top-3 Train Accuracy: {top3_train_acc:.4f}")
        print(f"Precision: {train_precision_dict}")
        print(f"Recall: {train_recall_dict}")

        # return the epoch number with the highest validation accuracy, for further steps on determining the early stopping.
        if X_val_tensor is not None and y_val_tensor is not None:
            best_epoch = val_accuracies_epoch.index(max(val_accuracies_epoch)) + 1
            print(f"Best Epoch (Validation Accuracy): {best_epoch}")
            print(f"Top-3 Val Accuracy: {top3_val_acc:.4f}")
            print(f"Precision: {val_precision_dict}")
            print(f"Recall: {val_recall_dict}")

            print(f"Top-3 Test Accuracy: {top3_test_acc:.4f}")
            print(f"Precision: {test_precision_dict}")
            print(f"Recall: {test_recall_dict}")

        elif X_test_tensor is not None and y_test_tensor is not None:
            best_epoch = test_accuracies_epoch.index(max(test_accuracies_epoch)) + 1
            print(f"Best Epoch (Test Accuracy): {best_epoch}")
            print(f"Top-3 Test Accuracy: {top3_test_acc:.4f}")
            print(f"Precision: {test_precision_dict}")
            print(f"Recall: {test_recall_dict}")

        # Build training history.
        training_history = {
            "train_accuracies_epoch": train_accuracies_epoch,
            "val_accuracies_epoch": val_accuracies_epoch if X_val_tensor is not None else [],
            "test_accuracies_epoch": test_accuracies_epoch if X_test_tensor is not None else [],
            "final_train_accuracy": final_train_acc,
            "training_model": self.model.__class__.__name__,
            "saved_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "best_epoch": best_epoch,
            "best_epoch accuracy": max(val_accuracies_epoch) if X_val_tensor is not None else max(test_accuracies_epoch) if X_test_tensor is not None else None,
            "train_precision": train_precision_dict,
            "train_recall": train_recall_dict,
            "top3_train_accuracy": top3_train_acc,
            "train_confusion_matrix": train_conf_matrix.tolist()
        }
        if final_val_acc is not None:
            training_history["final_validation_accuracy"] = final_val_acc
            training_history.update({
            "val_precision": val_precision_dict,
            "val_recall": val_recall_dict,
            "top3_val_accuracy": top3_val_acc,
        })
            training_history["val_confusion_matrix"] = val_conf_matrix.tolist()

        if final_test_acc is not None:
            training_history["final_test_accuracy"] = final_test_acc
            training_history.update({
            "test_precision": test_precision_dict,
            "test_recall": test_recall_dict,
            "top3_test_accuracy": top3_test_acc,
        })
            training_history["test_confusion_matrix"] = test_conf_matrix.tolist()
        
        if extra_info is not None:
            training_history.update(extra_info)
        
        with open(history_path, "w") as f:
            json.dump(training_history, f)
        
        return (train_accuracies_epoch, val_accuracies_epoch, test_accuracies_epoch)
    
    # This function computes the top-k accuracy.
    @staticmethod
    def compute_top_k_accuracy(logits, true_labels, k=3):
        top_k_preds = torch.topk(logits, k, dim=1).indices
        correct = top_k_preds.eq(true_labels.view(-1, 1).expand_as(top_k_preds))
        return correct.any(dim=1).float().mean().item()

    # This function evaluates the model on the provided data and returns the accuracy.
    def evaluate(self, X, y):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.long, device=self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)
            acc = accuracy_score(y_tensor.cpu().numpy(), predictions.cpu().numpy())
        return acc
