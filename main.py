# main.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from feature_extractor import FeatureExtractor
from classifier import MLP
from trainer import Trainer

def main(train_new=True):
    # Settings
    dataset_path = 'D:/DeepArch/arcDataset_shift_balanced'
    base_model_name = "ResNet152"  # For example, "ResNet152", "VGG16", etc.
    target_size = (224, 224)
    batch_size = 32
    epochs = 10
    hidden_size = 512
    history_path = "training_history.json"

    # Initialize feature extractor and load dataset.
    extractor = FeatureExtractor(base_model_name, target_size=target_size)
    image_data, labels = extractor.load_dataset(dataset_path)
    extractor.print_class_counts(labels)
    embeddings = extractor.get_embeddings_in_batches(image_data, batch_size=batch_size)
    extractor.save_embeddings(embeddings, output_folder="embeddings")
    print(f"Loaded {len(image_data)} images.")

    # Flatten embeddings.
    N, C, H, W = embeddings.shape
    embeddings_flat = embeddings.reshape(N, C * H * W)
    print(f"Flattened embeddings shape: {embeddings_flat.shape}")

    # Encode labels.
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Split the data.
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings_flat, labels_encoded, test_size=0.2, random_state=42
    )

    if train_new:
        # Create the classifier model.
        input_size = X_train.shape[1]
        output_size = len(np.unique(y_train))
        model = MLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

        # Build extra information to be saved with the training history.
        experiment_name = f"{base_model_name}_{target_size[0]}x{target_size[1]}_MLP"
        extra_info = {
            "experiment_name": experiment_name,
            "base_model": base_model_name,
            "image_size": target_size
        }

        # Train the model and save training history with extra information.
        trainer = Trainer(model=model, batch_size=batch_size, epochs=epochs)
        trainer.train(X_train, y_train, X_test, y_test, history_path=history_path, extra_info=extra_info)

        # Evaluate final accuracy.
        final_train_acc = trainer.evaluate(X_train, y_train)
        final_test_acc = trainer.evaluate(X_test, y_test)
        print(f"Final Train Accuracy: {final_train_acc * 100:.2f}%")
        print(f"Final Test Accuracy: {final_test_acc * 100:.2f}%")
    else:
        print("Skipping training; using saved training history.")

if __name__ == '__main__':
    main(train_new=True)
