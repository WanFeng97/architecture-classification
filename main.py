import numpy as np
from sklearn.preprocessing import LabelEncoder
from feature_extractor_torch import FeatureExtractor
from classifier import MLP
from trainer import Trainer
import torch
import os
import time
from sklearn.model_selection import train_test_split

# Set device to GPU if available. Initially, the tasks were done on CPU, but the training was extreamely slow compared to GPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the style classifier training function.
def train_style_classifier(train_dataset_path, test_dataset_path, base_model_name="resnet152",
                           target_size=(224, 224), batch_size=32, epochs=10, hidden_size=1024, history_path="style_training_history.json"):
    
    # It provides an option to load the embeddings from the disk, which is useful for debugging and testing.
    load_embeddings = False
    
    if not load_embeddings:
        # Load FeatureExtractor from feature_extractor.py
        # load the style dataset and save labels. 
        extractor = FeatureExtractor(base_model_name, target_size=target_size)
        train_image_data, train_labels = extractor.load_dataset(train_dataset_path)
        print("Style classification training set:")
        extractor.print_class_counts(train_labels)
        test_image_data, test_labels = extractor.load_dataset(test_dataset_path)
        print("Style classification test set:")
        extractor.print_class_counts(test_labels)

        os.makedirs("../labels", exist_ok=True)
        np.save('../labels/style_train_labels.npy', train_labels)
        np.save('../labels/style_test_labels.npy', test_labels)
        
        # CNN embeddings computation and saving, embeddings could be loaded from the disk for future use. 
        print("Generating training embeddings for style classification...")
        train_embeddings = extractor.get_embeddings_in_batches(train_image_data, batch_size=batch_size)
        print("Generating test embeddings for style classification...")
        test_embeddings = extractor.get_embeddings_in_batches(test_image_data, batch_size=batch_size)
        
        os.makedirs("../embeddings/style_train_embeddings", exist_ok=True)
        os.makedirs("../embeddings/style_test_embeddings", exist_ok=True)
        extractor.save_embeddings(train_embeddings, output_folder="../embeddings/style_train_embeddings")
        extractor.save_embeddings(test_embeddings, output_folder="../embeddings/style_test_embeddings")
        
        print(f"Flattened training embeddings shape: {train_embeddings.shape}")
        print(f"Flattened test embeddings shape: {test_embeddings.shape}")

        # Split the test embeddings into validation and test parts.
        test_embeddings, val_embeddings, test_labels, val_labels = train_test_split(
            test_embeddings, test_labels, test_size=0.25, random_state=42, stratify=test_labels
        )
        print(f"Validation embeddings shape: {val_embeddings.shape}")
        
        # Label encoding that converts categorical data into numerical format, which is required for training the classifier.
        label_encoder = LabelEncoder()
        all_labels = np.concatenate([train_labels, test_labels, val_labels])
        label_encoder.fit(all_labels)
        train_labels_encoded = label_encoder.transform(train_labels)
        test_labels_encoded = label_encoder.transform(test_labels)
        val_labels_encoded = label_encoder.transform(val_labels)
    else:
        # If load embeddings is true, get saved embeddings and labels from the disk.
        train_embeddings = np.load(f'../embeddings/style_train_embeddings/embeddings_{base_model_name}_{target_size[0]}x{target_size[1]}.npy')
        test_embeddings = np.load(f'../embeddings/style_test_embeddings/embeddings_{base_model_name}_{target_size[0]}x{target_size[1]}.npy')
        raw_train_labels = np.load('../labels/style_train_labels.npy')
        raw_test_labels = np.load('../labels/style_test_labels.npy')

        # Same split and encoding as above, but using the saved embeddings and labels.
        test_embeddings, val_embeddings, raw_test_labels, raw_val_labels = train_test_split(
            test_embeddings, raw_test_labels, test_size=0.25, random_state=42, stratify=raw_test_labels
        )

        label_encoder = LabelEncoder()
        all_labels = np.concatenate([raw_train_labels, raw_test_labels, raw_val_labels])
        label_encoder.fit(all_labels)
        train_labels_encoded = label_encoder.transform(raw_train_labels)
        test_labels_encoded = label_encoder.transform(raw_test_labels)
        val_labels_encoded = label_encoder.transform(raw_val_labels)
    
    # Create, train and save the style classifier.
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

    trainer = Trainer(model=style_classifier, learning_rate=1e-4, weight_decay=1e-4,
                    batch_size=batch_size, epochs=epochs)
    trainer.train(train_embeddings, train_labels_encoded, val_embeddings, val_labels_encoded,
                test_embeddings, test_labels_encoded,
                history_path=history_path, extra_info=extra_info, label_encoder=label_encoder)
    
    model_dir = os.path.dirname(history_path)
    os.makedirs(model_dir, exist_ok=True)
    style_model_path = os.path.join(model_dir, "style_classifier.pth")
    torch.save(style_classifier.state_dict(), style_model_path)
    print(f"Style classifier saved to: {style_model_path}")
            
    return style_classifier

# Compute softmax probabilities as style predictions
def get_style_predictions(model, embeddings, device):
    model.eval()
    with torch.no_grad():
        emb_tensor = torch.tensor(embeddings, dtype=torch.float32, device=device)
        logits = model(emb_tensor)
        probs = torch.softmax(logits, dim=1)
    return probs.cpu().numpy()

# Similar to the style classifier, but for the architect classifier, with the concatenation of style prediction probabilities.
def train_architect_classifier(style_classifier, architect_train_dataset_path, architect_test_dataset_path,
                               base_model_name, target_size, batch_size, epochs, hidden_size, history_path):
    extractor = FeatureExtractor(base_model_name, target_size=target_size)
    
    arch_train_images, arch_train_labels = extractor.load_dataset(architect_train_dataset_path)
    arch_test_images, arch_test_labels = extractor.load_dataset(architect_test_dataset_path)
    
    print("Architect training set:")
    extractor.print_class_counts(arch_train_labels)
    print("Architect test set:")
    extractor.print_class_counts(arch_test_labels)
    
    np.save('../labels/architect_train_labels.npy', arch_train_labels)
    np.save('../labels/architect_test_labels.npy', arch_test_labels)
    
    print("Generating architect training embeddings...")
    train_embeddings_arch = extractor.get_embeddings_in_batches(arch_train_images, batch_size=batch_size)
    print("Generating architect test embeddings...")
    test_embeddings_arch = extractor.get_embeddings_in_batches(arch_test_images, batch_size=batch_size)
    
    os.makedirs("../embeddings/architect_train_embeddings", exist_ok=True)
    os.makedirs("../embeddings/architect_test_embeddings", exist_ok=True)
    extractor.save_embeddings(train_embeddings_arch, output_folder="../embeddings/architect_train_embeddings")
    extractor.save_embeddings(test_embeddings_arch, output_folder="../embeddings/architect_test_embeddings")

    train_style_preds = get_style_predictions(style_classifier, train_embeddings_arch, device)
    test_style_preds = get_style_predictions(style_classifier, test_embeddings_arch, device)
    
    # Concatenate the original CNN embeddings with style predictions
    train_arch_embeddings = np.concatenate([train_embeddings_arch, train_style_preds], axis=1)
    test_arch_embeddings = np.concatenate([test_embeddings_arch, test_style_preds], axis=1)
    
    label_encoder_arch = LabelEncoder()
    all_arch_labels = np.concatenate([arch_train_labels, arch_test_labels])
    label_encoder_arch.fit(all_arch_labels)
    train_arch_labels_enc = label_encoder_arch.transform(arch_train_labels)
    test_arch_labels_enc = label_encoder_arch.transform(arch_test_labels)
    
    input_size_arch = train_arch_embeddings.shape[1]
    output_size_arch = len(np.unique(train_arch_labels_enc))  # Expected to be 17.
    architect_classifier = MLP(input_size=input_size_arch, hidden_size=hidden_size,
                               output_size=output_size_arch, dropout_rate=0.5).to(device)
    
    extra_info = {
        "phase": "architect classification",
        "concat_features": "CNN embeddings concatenated with style prediction probabilities"
    }
    
    # Train and save the architect classifier.
    trainer_arch = Trainer(model=architect_classifier, learning_rate=5e-5,
                           weight_decay=1e-4, batch_size=batch_size, epochs=epochs)
    trainer_arch.train(train_arch_embeddings, train_arch_labels_enc,
                       X_test=test_arch_embeddings, y_test=test_arch_labels_enc,
                       history_path=history_path, extra_info=extra_info, label_encoder=label_encoder_arch)
    
    model_dir = os.path.dirname(history_path)
    os.makedirs(model_dir, exist_ok=True)
    architect_model_path = os.path.join(model_dir, "architect_classifier.pth")
    torch.save(architect_classifier.state_dict(), architect_model_path)
    print(f"Architect classifier saved to: {architect_model_path}")
    
    return architect_classifier

def main():
    # The training phases can be controlled, decide if the style and architect phase, or both, should be trained
    train_style = True
    train_architect = True
    
    # Paths for the style dataset (for training the style classifier).
    style_train_dataset_path = '../data/augmented_data/style_classification/style_train_augmented'
    style_test_dataset_path = '../data/augmented_data/style_classification/style_test'

    # Paths for the architect subset (with 11 or 17 architects, the default is 11 classes architect data).
    architect_train_dataset_path = '../data/augmented_data/architect_classification/architect_11classes/architect_train_augmented'
    architect_test_dataset_path = '../data/augmented_data/architect_classification/architect_11classes/architect_test'
    
    # change the base model name for vgg16, vgg11, resnet152, resnet50, resnext101_64x4d, mobilenet_v3_large, densenet201
    base_model_name = "mobilenet_v3_large"
    target_size = (224, 224)
    batch_size = 32   
    style_epochs = 10
    architect_epochs = 10
    hidden_size = 1024
    input_size = 2048
    output_size = 25

    # create a folder with the current base_model_name and datetime, to save the training history and models.
    current_time = time.strftime("%Y%m%d-%H%M%S")
    folder_name = os.path.join(base_model_name, current_time)
    os.makedirs(folder_name, exist_ok=True)
    
    # Phase 1: Train (and save) the style classifier.
    if train_style:
        style_classifier = train_style_classifier(
            style_train_dataset_path, 
            style_test_dataset_path,
            base_model_name, 
            target_size, 
            batch_size, 
            style_epochs, 
            hidden_size, 
            history_path=os.path.join(folder_name, "style_training_history.json"),
        )
    else:
        # Load the saved style classifier if already trained, if style training is not needed.
        sample_embedding = np.load(f'../embeddings/style_train_embeddings/embeddings_{base_model_name}_{target_size[0]}x{target_size[1]}.npy')
        input_size = sample_embedding.shape[1]
        style_classifier = MLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
        # Load the model to predict the style of the images, and concatenate the style prediction probabilities with the architect CNN embeddings.The path should be specified based on the different base models.
        style_classifier.load_state_dict(torch.load("../training_history/mobilenet_v3_large/mobilenet_v3_large_style_best_epoch/style_classifier.pth"))
        style_classifier.eval()
    
    # Phase 2: Train the architect classifier using the architect subset.
    if train_architect:
        train_architect_classifier(
            style_classifier,
            architect_train_dataset_path,
            architect_test_dataset_path,
            base_model_name, 
            target_size, 
            batch_size, 
            architect_epochs, 
            hidden_size, 
            history_path=os.path.join(folder_name, "architect_training_history.json"),
        )

if __name__ == '__main__':
    main()
