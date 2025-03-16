# feature_extractor.py
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import nnabla as nn
import nnabla.models.imagenet as models 
from collections import Counter

class FeatureExtractor:
    def __init__(self, model_name, target_size=(64, 64)):
        self.target_size = target_size
        self.model_name = model_name
        self.model = self._load_model(model_name)
    
    def _load_model(self, model_name):
        try:
            # Get the model from nnabla
            model_class = getattr(models, model_name)
        except AttributeError:
            raise ValueError(f"Model {model_name} not found in nnabla.models.imagenet.")
        return model_class()

    def preprocess_image(self, image_path):
        # Open, resize, and convert an image to a channels-first numpy array.
        try:
            img = Image.open(image_path)
        except Exception as e:
            print(f"Error opening image {image_path}: {e}. Skipping.")
            return None
        
        img = img.resize(self.target_size)
        img_array = np.array(img)
        
        # Check if the image is RGB (3 channels)
        if len(img_array.shape) != 3 or img_array.shape[-1] != 3:
            print(f"Warning: Skipping image {image_path} as it doesn't have 3 channels.")
            return None
        
        # Convert from (H, W, C) to (C, H, W)
        img_array = np.transpose(img_array, (2, 0, 1))
        return img_array

    def load_dataset(self, dataset_path):
        # Load dataset images and labels from a directory structure.
        image_data = []
        labels = []
        styles = os.listdir(dataset_path)
        for style in styles:
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
        # Generate embeddings for the image data in mini-batches.
        embeddings_list = []
        for i in tqdm(range(0, len(image_data), batch_size), desc="Generating Embeddings", ncols=100):
            batch = image_data[i:i+batch_size]
            x = nn.Variable(batch.shape)
            x.d = batch
            
            # Forward pass through the model until the pooling layer
            embeddings = self.model(x, training=False, use_up_to='pool')
            embeddings.forward()
            embeddings_list.append(embeddings.d)
            del embeddings  # free memory
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
        and the image size (e.g., embeddings_ResNet50_64x64.npy).
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        file_name = f"embeddings_{self.model_name}_{self.target_size[0]}x{self.target_size[1]}.npy"
        file_path = os.path.join(output_folder, file_name)
        np.save(file_path, embeddings)
        print(f"Embeddings saved to: {file_path}")
