import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.models as models
from collections import Counter
import torchvision.transforms as transforms

# Set device to GPU if available. Initially, the tasks were done on CPU, but the training was extreamely slow compared to GPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This class is used to extract features from images using a pretrained model from torchvision.
class FeatureExtractor:
    def __init__(self, model_name, target_size=(64, 64)):
        self.target_size = target_size
        self.model_name = model_name
        self.model = self._load_model(model_name)
        self.model.to(device)
        self.model.eval()

    # load the model from torchvision.models
    def _load_model(self, model_name):
        try:
            model_class = getattr(models, model_name)
        except AttributeError:
            raise ValueError(f"Model {model_name} not found in torchvision.models.")

        # Here it gets pretrained weights for different architectures.
        if model_name.lower().startswith("resnet"):
            model = model_class(weights='IMAGENET1K_V2')
            # remove fc layer, only need the feature extractor part, not the final class probability from the model
            modules = list(model.children())[:-1]  
            model = nn.Sequential(*modules)
        
        elif model_name.lower().startswith("resnext"):
            model = model_class(weights='IMAGENET1K_V1')
            modules = list(model.children())[:-1]  
            model = nn.Sequential(*modules)

        elif model_name.lower().startswith("vgg"):
            # No need to remove last layer for vgg. The feature extractor and classifier are already separated internally inside torchvision.models.vgg
            model = model_class(weights='IMAGENET1K_V1')
            model = model.features

        elif model_name.lower().startswith("densenet201"):
            model = model_class(weights='IMAGENET1K_V1')
            modules = list(model.children())[:-1]  
            model = nn.Sequential(*modules)

        elif model_name.lower().startswith("mobilenet_v3"):
            model = model_class(weights='IMAGENET1K_V1')
            modules = list(model.children())[:-1]  
            model = nn.Sequential(*modules)

        else:
            raise ValueError(f"Model architecture '{model_name}' is not supported.")

        return model

    # Images are processed to the required size and normalizes it.
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
            img_tensor = self.transform(img) 
            return img_tensor.cpu().numpy()
        else:
            img = img.resize(self.target_size)
            img_array = np.array(img)
            if len(img_array.shape) != 3 or img_array.shape[-1] != 3:
                print(f"Warning: Skipping image {image_path} as it doesn't have 3 channels.")
                return None
            img_array = np.transpose(img_array, (2, 0, 1)).astype(np.float32) / 255.0
            return img_array

    # Load the dataset from the specified path. The dataset has been argumented and stored by data_augmentation_split.py. tqdm is used to show the progress bar.
    def load_dataset(self, dataset_path):
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
    
    # Generate embeddings in mini-batches using PyTorch on GPU.
    def get_embeddings_in_batches(self, image_data, batch_size=32):
        embeddings_list = []
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(image_data), batch_size), desc="Generating Embeddings", ncols=100):
                batch = image_data[i:i+batch_size]
                batch_tensor = torch.tensor(batch, dtype=torch.float32).to(device)
                outputs = self.model(batch_tensor)
                outputs = outputs.view(outputs.size(0), -1)
                embeddings_list.append(outputs.cpu().numpy())
        return np.concatenate(embeddings_list, axis=0)
    
    # This function is used to print the class counts in the dataset.
    def print_class_counts(self, labels):
        counts = Counter(labels)
        print("Class distribution:")
        for label, count in counts.items():
            print(f"{label}: {count} images")
    
    # save embeddings for future use.
    def save_embeddings(self, embeddings, output_folder="."):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        file_name = f"embeddings_{self.model_name}_{self.target_size[0]}x{self.target_size[1]}.npy"
        file_path = os.path.join(output_folder, file_name)
        np.save(file_path, embeddings)
        print(f"Embeddings saved to: {file_path}")