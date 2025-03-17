# plotter.py
import json
import matplotlib.pyplot as plt

def plot_epoch_accuracy(history_path="training_history.json"):
    with open(history_path, "r") as f:
        history = json.load(f)
    
    train_accs = history["train_accuracies_epoch"]
    test_accs = history["test_accuracies_epoch"]
    final_train_acc = history["final_train_accuracy"]
    final_test_acc = history["final_test_accuracy"]
    model_name = history.get("training_model", "Unknown Model")
    epochs = range(1, len(train_accs) + 1)

    plt.figure(figsize=(10, 6))
    plt.ylim(0, 1)
    line1, = plt.plot(epochs, train_accs, label="Train Accuracy", color="#FFBC24")
    line2, = plt.plot(epochs, test_accs, label="Test Accuracy", color="#086CB4")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Testing Accuracy Over Epochs")
    plt.grid(True)
    legend_title = (
        f"Model: {model_name}\n"
        f"Final Train Acc: {final_train_acc * 100:.2f}%\n"
        f"Final Test Acc: {final_test_acc * 100:.2f}%"
    )
    legend = plt.legend(handles=[line1, line2], loc='upper left', title=legend_title, prop={'size': 12})
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.7)
    plt.setp(legend.get_title(), fontsize=12)
    plt.show()

def plot_batch_accuracy(history_path="training_history.json"):
    with open(history_path, "r") as f:
        history = json.load(f)
    
    train_batch_accs = history["train_accuracies_batch"]
    test_batch_accs = history["test_accuracies_batch"]
    model_name = history.get("training_model", "Unknown Model")
    # x-axis as batch iterations
    batches = range(1, len(train_batch_accs) + 1)

    plt.figure(figsize=(10, 6))
    plt.ylim(0, 1)
    line1, = plt.plot(batches, train_batch_accs, label="Train Accuracy (batch)", color="#FF5733")
    line2, = plt.plot(batches, test_batch_accs, label="Test Accuracy (batch)", color="#33A1FF")
    plt.xlabel("Batch iteration")
    plt.ylabel("Accuracy")
    plt.title("Training and Testing Accuracy Over Batches")
    plt.grid(True)
    legend_title = f"Model: {model_name}"
    legend = plt.legend(handles=[line1, line2], loc='upper left', title=legend_title, prop={'size': 12})
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.7)
    plt.setp(legend.get_title(), fontsize=12)
    plt.show()

if __name__ == "__main__":
    # Plot epoch-level accuracy.
    plot_epoch_accuracy()
    # Plot batch-level accuracy.
    plot_batch_accuracy()
