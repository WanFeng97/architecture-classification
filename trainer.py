# trainer.py
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import json
from datetime import datetime

class Trainer:
    def __init__(self, model, learning_rate=0.001, weight_decay=0.0001, batch_size=32, epochs=10):
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
    
    def train(self, X_train, y_train, X_test, y_test, history_path=f"training_history.json", extra_info=None):
        # Convert numpy arrays to torch tensors.
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        train_data = TensorDataset(X_train_tensor, y_train_tensor)
        test_data = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        train_accuracies_epoch = []
        test_accuracies_epoch = []

        for epoch in range(self.epochs):
            self.model.train()
            print(f"Epoch {epoch+1}/{self.epochs}")
            with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}", ncols=100) as pbar:
                for X_batch, y_batch in train_loader:
                    self.optimizer.zero_grad()
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)
                    loss.backward()
                    self.optimizer.step()
                    pbar.update(1)

            # Evaluate accuracy at the end of the epoch.
            self.model.eval()
            with torch.no_grad():
                train_outputs = self.model(X_train_tensor)
                train_pred = torch.argmax(train_outputs, dim=1)
                train_acc = accuracy_score(y_train_tensor, train_pred)

                test_outputs = self.model(X_test_tensor)
                test_pred = torch.argmax(test_outputs, dim=1)
                test_acc = accuracy_score(y_test_tensor, test_pred)

            train_accuracies_epoch.append(train_acc)
            test_accuracies_epoch.append(test_acc)
            print(f"Train Accuracy: {train_acc:.4f} | Test Accuracy: {test_acc:.4f}")

        # After training, compute the final accuracies.
        self.model.eval()
        with torch.no_grad():
            train_outputs = self.model(X_train_tensor)
            train_pred = torch.argmax(train_outputs, dim=1)
            final_train_acc = accuracy_score(y_train_tensor, train_pred)

            test_outputs = self.model(X_test_tensor)
            test_pred = torch.argmax(test_outputs, dim=1)
            final_test_acc = accuracy_score(y_test_tensor, test_pred)

        # Build the training history dictionary with extra information.
        training_history = {
            "train_accuracies": train_accuracies_epoch,
            "test_accuracies": test_accuracies_epoch,
            "final_train_accuracy": final_train_acc,
            "final_test_accuracy": final_test_acc,
            "training_model": self.model.__class__.__name__,
            "saved_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        # Merge any extra info passed in.
        if extra_info is not None:
            training_history.update(extra_info)

        # Save the training history.
        with open(history_path, "w") as f:
            json.dump(training_history, f)
        
        return train_accuracies_epoch, test_accuracies_epoch

    def evaluate(self, X, y):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)
            acc = accuracy_score(y_tensor, predictions)
        return acc
