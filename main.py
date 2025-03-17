# main.py
import numpy as np
from sklearn.preprocessing import LabelEncoder
from feature_extractor_torch import FeatureExtractor
from classifier import MLP
from trainer import Trainer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(train_new=True):
    # Settings: update these paths as needed.
    train_dataset_path = 'D:/DeepArch/arcDataset_train_augmented'
    test_dataset_path = 'D:/DeepArch/arcDataset_test'
    base_model_name = "resnet152"  # e.g., "ResNet152", "VGG16", etc.
    target_size = (224, 224)
    batch_size = 32
    epochs = 10
    hidden_size = 1024
    history_path = "training_history.json"
    load_embeddings = False

    if not load_embeddings:
        # Initialize feature extractor.
        extractor = FeatureExtractor(base_model_name, target_size=target_size)
        
        # Load training dataset.
        train_image_data, train_labels = extractor.load_dataset(train_dataset_path)
        print("Training set:")
        extractor.print_class_counts(train_labels)
        
        # Load test dataset.
        test_image_data, test_labels = extractor.load_dataset(test_dataset_path)
        print("Test set:")
        extractor.print_class_counts(test_labels)

        np.save('labels/train_labels.npy', train_labels)
        np.save('labels/test_labels.npy', test_labels)
    
        # Compute embeddings for training and test sets separately.
        print("Generating training embeddings...")
        train_embeddings = extractor.get_embeddings_in_batches(train_image_data, batch_size=batch_size)
        print("Generating test embeddings...")
        test_embeddings = extractor.get_embeddings_in_batches(test_image_data, batch_size=batch_size)
        
        # Optionally, save the training embeddings.
        extractor.save_embeddings(train_embeddings, output_folder="train_embeddings")
        extractor.save_embeddings(test_embeddings, output_folder="test_embeddings")
    else:
        train_embeddings = np.load(f'train_embeddings\embeddings_{base_model_name}_{target_size[0]}x{target_size[1]}.npy')
        test_embeddings = np.load(f'test_embeddings\embeddings_{base_model_name}_{target_size[0]}x{target_size[1]}.npy')
        train_labels = np.load('labels/train_labels.npy')
        test_labels = np.load('labels/test_labels.npy')

    # Flatten embeddings.
    # N, C, H, W = train_embeddings.shape
    train_embeddings_flat = train_embeddings
    # .reshape(N, C * H * W)
    print(f"Flattened training embeddings shape: {train_embeddings_flat.shape}")
    
    # N_test, C_test, H_test, W_test = test_embeddings.shape
    test_embeddings_flat = test_embeddings
    # .reshape(N_test, C_test * H_test * W_test)
    print(f"Flattened test embeddings shape: {test_embeddings_flat.shape}")

    # Encode labels: fit on the combined set for consistency.
    label_encoder = LabelEncoder()
    all_labels = np.concatenate([train_labels, test_labels])
    label_encoder.fit(all_labels)
    train_labels_encoded = label_encoder.transform(train_labels)
    test_labels_encoded = label_encoder.transform(test_labels)
    
    if train_new:
        # Create the classifier model.
        input_size = train_embeddings_flat.shape[1]
        output_size = len(np.unique(train_labels_encoded))
        model = MLP(
            input_size=input_size, 
            hidden_size=hidden_size, 
            output_size=output_size,
            dropout_rate=0.5,
        ).to(device)
    
        # Build extra information to be saved with the training history.
        experiment_name = f"{base_model_name}_{target_size[0]}x{target_size[1]}_MLP"
        extra_info = {
            "experiment_name": experiment_name,
            "base_model": base_model_name,
            "image_size": target_size
        }
    
        # Train the model on the training set and evaluate on the test set.
        trainer = Trainer(
            model=model, 
            learning_rate=1e-4, 
            weight_decay=1e-4, 
            batch_size=batch_size, 
            epochs=epochs
        )
        trainer.train(train_embeddings_flat, train_labels_encoded,
                      test_embeddings_flat, test_labels_encoded,
                      history_path=history_path, extra_info=extra_info)
    
        # # Evaluate final accuracy.
        # final_train_acc = trainer.evaluate(train_embeddings_flat, train_labels_encoded)
        # final_test_acc = trainer.evaluate(test_embeddings_flat, test_labels_encoded)
        # print(f"Final Train Accuracy: {final_train_acc * 100:.2f}%")
        # print(f"Final Test Accuracy: {final_test_acc * 100:.2f}%")
    else:
        print("Skipping training; using saved training history.")

if __name__ == '__main__':
    main(train_new=True)