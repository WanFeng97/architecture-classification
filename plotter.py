# plotter.py
import json
import matplotlib.pyplot as plt

def plot_training_history(history_path="training_history.json"):
    # Load training history from file.
    with open(history_path, "r") as f:
        history = json.load(f)
    
    train_accs = history["train_accuracies"]
    test_accs = history["test_accuracies"]
    final_train_acc = history["final_train_accuracy"]
    final_test_acc = history["final_test_accuracy"]
    model_name = history.get("experiment_name", "Unknown Model")
    epochs = range(1, len(train_accs) + 1)

    # Plot the accuracy curves.
    f, ax = plt.subplots(1, figsize=(10, 6))
    ax.set_ylim(ymin=0)
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

if __name__ == "__main__":
    plot_training_history()
