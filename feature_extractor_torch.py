# feature_extractor.py
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.models as models
from collections import Counter
import torchvision.transforms as transforms

# Set device to GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FeatureExtractor:
    def __init__(self, model_name, target_size=(64, 64)):
        self.target_size = target_size
        self.model_name = model_name
        self.model = self._load_model(model_name)
        self.model.to(device)
        self.model.eval()  # Set model to evaluation mode.
    
    def _load_model(self, model_name):
        try:
            # Get the model from torchvision.models
            model_class = getattr(models, model_name)
        except AttributeError:
            raise ValueError(f"Model {model_name} not found in torchvision.models.")
        # Load pretrained weights.
        model = model_class(weights='IMAGENET1K_V2')
        # For ResNet, remove the final fully connected layer.
        if model_name.lower().startswith("resnet"):
            modules = list(model.children())[:-1]
            model = nn.Sequential(*modules)
        # For VGG might want to use only features.
        elif model_name.lower().startswith("vgg"):
            model = model.features
        return model

    def preprocess_image(self, image_path):
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),  # converts to [0,1] and (C,H,W)
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
        ])
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {image_path}: {e}. Skipping.")
            return None
        if hasattr(self, 'transform') and self.transform is not None:
            img_tensor = self.transform(img)  # This returns a tensor already.
            # Convert to numpy if needed (or change get_embeddings_in_batches accordingly).
            return img_tensor.cpu().numpy()
        else:
            img = img.resize(self.target_size)
            img_array = np.array(img)
            if len(img_array.shape) != 3 or img_array.shape[-1] != 3:
                print(f"Warning: Skipping image {image_path} as it doesn't have 3 channels.")
                return None
            img_array = np.transpose(img_array, (2, 0, 1)).astype(np.float32) / 255.0
            return img_array

    def load_dataset(self, dataset_path):
        # Load dataset images and labels from a directory structure.
        image_data = []
        labels = []
        styles = os.listdir(dataset_path)
        for style in tqdm(styles):
            style_path = os.path.join(dataset_path, style)
            if os.path.isdir(style_path):
                image_files = [f for f in os.listdir(style_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
                for image_file in image_files:
                    image_path = os.path.join(style_path, image_file)
                    img = self.preprocess_image(image_path)
                    if img is None:
                        continue
                    image_data.append(img)
                    labels.append(style)
        return np.array(image_data), np.array(labels)

    def get_embeddings_in_batches(self, image_data, batch_size=32):
        # Generate embeddings in mini-batches using PyTorch on GPU.
        embeddings_list = []
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(image_data), batch_size), desc="Generating Embeddings", ncols=100):
                batch = image_data[i:i+batch_size]
                # Convert batch to torch tensor and move to GPU.
                batch_tensor = torch.tensor(batch, dtype=torch.float32).to(device)
                # Forward pass through the model.
                outputs = self.model(batch_tensor)
                # For a ResNet, outputs are (batch, features, 1, 1). Flatten to (batch, features).
                outputs = outputs.view(outputs.size(0), -1)
                embeddings_list.append(outputs.cpu().numpy())
        return np.concatenate(embeddings_list, axis=0)
    
    def print_class_counts(self, labels):
        """
        Print how many images there are for each class, based on the provided labels.
        """
        counts = Counter(labels)
        print("Class distribution:")
        for label, count in counts.items():
            print(f"{label}: {count} images")
    
    def save_embeddings(self, embeddings, output_folder="."):
        """
        Save the embeddings to a .npy file. The filename will include the base model
        and the image size (e.g., embeddings_ResNet152_64x64.npy).
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        file_name = f"embeddings_{self.model_name}_{self.target_size[0]}x{self.target_size[1]}.npy"
        file_path = os.path.join(output_folder, file_name)
        np.save(file_path, embeddings)
        print(f"Embeddings saved to: {file_path}")