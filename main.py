import numpy as np
from sklearn.preprocessing import LabelEncoder
from feature_extractor_torch import FeatureExtractor
from classifier import MLP
from trainer import Trainer
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_style_classifier(train_dataset_path, test_dataset_path, base_model_name="resnet152",
                           target_size=(224, 224), batch_size=32, epochs=10, hidden_size=1024):
    load_embeddings = True
    if not load_embeddings:
        # Load FeatureExtractor from feature_extractor.py
        # load the style dataset
        extractor = FeatureExtractor(base_model_name, target_size=target_size)
        train_image_data, train_labels = extractor.load_dataset(train_dataset_path)
        print("Style classification training set:")
        extractor.print_class_counts(train_labels)
        test_image_data, test_labels = extractor.load_dataset(test_dataset_path)
        print("Style classification test set:")
        extractor.print_class_counts(test_labels)
        
        # Style labels could be saved for future use
        os.makedirs("labels", exist_ok=True)
        np.save('labels/train_labels.npy', train_labels)
        np.save('labels/test_labels.npy', test_labels)
        
        # CNN embeddings computation
        print("Generating training embeddings for style classification...")
        train_embeddings = extractor.get_embeddings_in_batches(train_image_data, batch_size=batch_size)
        print("Generating test embeddings for style classification...")
        test_embeddings = extractor.get_embeddings_in_batches(test_image_data, batch_size=batch_size)
        
        # CNN embeddings could be saved for later usage, due to the time-consuming process
        os.makedirs("train_embeddings", exist_ok=True)
        os.makedirs("test_embeddings", exist_ok=True)
        extractor.save_embeddings(train_embeddings, output_folder="train_embeddings")
        extractor.save_embeddings(test_embeddings, output_folder="test_embeddings")
        
        print(f"Flattened training embeddings shape: {train_embeddings.shape}")
        print(f"Flattened test embeddings shape: {test_embeddings.shape}")
        
        # Label encoding that converts categorical data into numerical format
        label_encoder = LabelEncoder()
        all_labels = np.concatenate([train_labels, test_labels])
        label_encoder.fit(all_labels)
        train_labels_encoded = label_encoder.transform(train_labels)
        test_labels_encoded = label_encoder.transform(test_labels)
    else:
        # Load saved embeddings and labels
        train_embeddings = np.load(f'train_embeddings/embeddings_{base_model_name}_{target_size[0]}x{target_size[1]}.npy')
        test_embeddings = np.load(f'test_embeddings/embeddings_{base_model_name}_{target_size[0]}x{target_size[1]}.npy')
        raw_train_labels = np.load('labels/train_labels.npy')
        raw_test_labels = np.load('labels/test_labels.npy')

        label_encoder = LabelEncoder()
        all_labels = np.concatenate([raw_train_labels, raw_test_labels])
        label_encoder.fit(all_labels)
        train_labels_encoded = label_encoder.transform(raw_train_labels)
        test_labels_encoded = label_encoder.transform(raw_test_labels)
    
    # Create the style classifier.
    input_size = train_embeddings.shape[1]
    output_size = len(np.unique(train_labels_encoded))
    style_classifier = MLP(input_size=input_size, hidden_size=hidden_size,
                        output_size=output_size, dropout_rate=0.5).to(device)
        
    experiment_name = f"{base_model_name}_{target_size[0]}x{target_size[1]}_Style_MLP"
    extra_info = {
        "experiment_name": experiment_name,
        "base_model": base_model_name,
        "image_size": target_size,
        "phase": "style classification"
    }

    # Train the style classifier.
    trainer = Trainer(model=style_classifier, learning_rate=1e-4, weight_decay=1e-4,
                    batch_size=batch_size, epochs=epochs)
    trainer.train(train_embeddings, train_labels_encoded,
                test_embeddings, test_labels_encoded,
                history_path="style_training_history.json", extra_info=extra_info)
    
    # Save the trained style classifier.
    os.makedirs("models", exist_ok=True)
    style_model_path = os.path.join("models", "style_classifier.pth")
    torch.save(style_classifier.state_dict(), style_model_path)
    print(f"Style classifier saved to: {style_model_path}")
            
    return style_classifier

def get_style_predictions(model, embeddings, device):
    """
    Given a trained style classifier and CNN embeddings,
    compute softmax probabilities as style predictions.
    """
    model.eval()
    with torch.no_grad():
        emb_tensor = torch.tensor(embeddings, dtype=torch.float32, device=device)
        logits = model(emb_tensor)
        probs = torch.softmax(logits, dim=1)
    return probs.cpu().numpy()

def train_architect_classifier(style_classifier, architect_train_dataset_path, architect_test_dataset_path,
                               base_model_name, target_size, batch_size, epochs, hidden_size):
    extractor = FeatureExtractor(base_model_name, target_size=target_size)
    
    arch_train_images, arch_train_labels = extractor.load_dataset(architect_train_dataset_path)
    arch_test_images, arch_test_labels = extractor.load_dataset(architect_test_dataset_path)
    
    print("Architect training set:")
    extractor.print_class_counts(arch_train_labels)
    print("Architect test set:")
    extractor.print_class_counts(arch_test_labels)
    
    # Save architect labels, which will benifit the future use
    np.save('labels/architect_train_labels.npy', arch_train_labels)
    np.save('labels/architect_test_labels.npy', arch_test_labels)
    
    # CNN embeddings for the architect images
    print("Generating architect training embeddings...")
    train_embeddings_arch = extractor.get_embeddings_in_batches(arch_train_images, batch_size=batch_size)
    print("Generating architect test embeddings...")
    test_embeddings_arch = extractor.get_embeddings_in_batches(arch_test_images, batch_size=batch_size)
    
    # Generate style prediction probabilities using the saved style classifier
    train_style_preds = get_style_predictions(style_classifier, train_embeddings_arch, device)
    test_style_preds = get_style_predictions(style_classifier, test_embeddings_arch, device)
    
    # Concatenate the original CNN embeddings with style predictions
    train_arch_embeddings = np.concatenate([train_embeddings_arch, train_style_preds], axis=1)
    test_arch_embeddings = np.concatenate([test_embeddings_arch, test_style_preds], axis=1)
    
    # Encode architect labels
    label_encoder_arch = LabelEncoder()
    all_arch_labels = np.concatenate([arch_train_labels, arch_test_labels])
    label_encoder_arch.fit(all_arch_labels)
    train_arch_labels_enc = label_encoder_arch.transform(arch_train_labels)
    test_arch_labels_enc = label_encoder_arch.transform(arch_test_labels)
    
    # Architect classifier model
    input_size_arch = train_arch_embeddings.shape[1]
    output_size_arch = len(np.unique(train_arch_labels_enc))  # Expected to be 17.
    architect_classifier = MLP(input_size=input_size_arch, hidden_size=hidden_size,
                               output_size=output_size_arch, dropout_rate=0.5).to(device)
    
    extra_info = {
        "phase": "architect classification",
        "concat_features": "CNN embeddings concatenated with style prediction probabilities"
    }
    
    # Train the architect classifier.
    trainer_arch = Trainer(model=architect_classifier, learning_rate=5e-5,
                           weight_decay=1e-4, batch_size=batch_size, epochs=epochs)
    trainer_arch.train(train_arch_embeddings, train_arch_labels_enc,
                       test_arch_embeddings, test_arch_labels_enc,
                       history_path="architect_training_history.json", extra_info=extra_info)
    
    # Save the architect classifier.
    architect_model_path = os.path.join("models", "architect_classifier.pth")
    torch.save(architect_classifier.state_dict(), architect_model_path)
    print(f"Architect classifier saved to: {architect_model_path}")
    
    return architect_classifier

def main():
    # Control training phases.
    train_style = False
    train_architect = True
    
    # Paths for the style dataset (for training the style classifier).
    style_train_dataset_path = 'D:/DeepArch/arcDataset_train_augmented'
    style_test_dataset_path = 'D:/DeepArch/arcDataset_test'
    
    # Paths for the architect subset (with 17 architects).
    architect_train_dataset_path = 'D:/DeepArch/architect_train_augmented'
    architect_test_dataset_path = 'D:/DeepArch/architect_test'
    
    base_model_name = "resnet152"
    target_size = (224, 224)
    batch_size = 32
    style_epochs = 10
    architect_epochs = 10
    hidden_size = 1024
    input_size = 2048
    output_size = 25

    
    # Phase 1: Train (and save) the style classifier.
    if train_style:
        style_classifier = train_style_classifier(
            style_train_dataset_path, style_test_dataset_path,
            base_model_name, target_size, batch_size, style_epochs, hidden_size
        )
    else:
        # Load the saved style classifier if already trained.
        style_classifier = MLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
        style_classifier.load_state_dict(torch.load("models/style_classifier.pth"))
        style_classifier.eval()
    
    # Phase 2: Train the architect classifier using the architect subset.
    if train_architect:
        train_architect_classifier(style_classifier,
                                   architect_train_dataset_path,
                                   architect_test_dataset_path,
                                   base_model_name, target_size, batch_size, architect_epochs, hidden_size)

if __name__ == '__main__':
    main()
