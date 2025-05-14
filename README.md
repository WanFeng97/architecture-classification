# architecture-classification
Artefact for Dissertation Project: Architects and Architectural Styles Classification for Education and Historic Preservation Efforts using Image Recognition Techniques

1. Overview
This study aims to investigate using pre-trained neural networks for accurate, efficient, and contextually insightful style and architect classification. The developed model targets architecture practitioners working in education and heritage preservation, especially those operating with limited computational resources. Ultimately, the project seeks to automate and accelerate the architectural analysis process, contributing to the efficient preservation of cultural heritage and enhancing educational tools for architects and historians.

2. Artefact Directory Structure
File	Purpose
embeddings	Stores computed CNN embeddings.
labels	Stores style and architect label files gained form datasets, for training usage. 
models	Contains the trained style and architect classifier model for GUI. PTH files can be loaded to continue training or to make predictions.
3. Code Files Description
Path: artefact/code
app.py	GUI application built using tkinter.
classifier.py	Defines the MLP classifier used for predicting architectural styles and architects.
data_augmentation_split.py	Performs dataset splitting into training and test sets, and applies data augmentation techniques.
feature_extractor_torch.py	Extracts image features using pre-trained CNN models from PyTorch's torchvision library.
main.py	Main script to initialize the feature extractor and training on style and architect classifiers.
plotter.ipynb	Notebook used for visualizing results and performance metrics.
trainer.py	Trains and evaluates the MLP classifiers, providing detailed accuracy metrics.

3.1 Code Dependencies
	torch 1.12.0, torchvision 0.13.0
	numpy 1.26.4
	scikit-learn 1.6.1
	Pillow (PIL) 11.1.0
	tkinter 8.6.11
	tqdm 4.62.3
3.2 Environment Setup
A complete environment, including all required packages (e.g., PyTorch, torchvision, scikit-learn, Pillow, etc.), should support the reproduce of the result 
3.3 Usage
3.3.1 Data Augmentation
Data Augmentation performs class-preserving train/test splitting, and applies random visual transformations to increase dataset variability and generalization capacity. data_augmentation_split.py handles both dataset splitting and augmentation.
How to use:
1.	Specify the input raw data path and the output folder
2.	Run data_augmentation_split.py 
3.3.2 Phase 1: Feature Extractor
The project uses a pretrained CNN (vgg16, vgg11, resnet152, resnet50, resnext101_64x4d, mobilenet_v3_large, densenet201) to extract deep feature embeddings from images. The FeatureExtractor class (from feature_extractor_torch.py) handles image loading, resizing, normalization, and embedding extraction. Embeddings are generated in batches and saved to disk for future reuse to speed up training. This feature extraction is triggered automatically by calling train_style_classifier() and train_architect_classifier() in main.py.
How to use: 
1.	launch main.py.
2.	Adjust which base model to use (base_model_name) and target image size (target_size) inside main.py under def main().
3.	To get new embeddings, run main.py with load_embeddings = False (default).
4.	To load the saved style embeddings instead of creating new ones, toggle load_embeddings = True inside train_style_classifier(), and specify the file path here: style_classifier.load_state_dict(torch.load("../training_history/mobilenet_v3_large/mobilenet_v3_large_style_best_epoch/style_classifier.pth")) in main.py
3.3.3 Phase 2-1: Training Models (Style)
The style classifier is trained to predict the architectural style of a building image based on its extracted deep feature embeddings.
	train_style_classifier() (in main.py) controls the full training pipeline for the style classifier.
	Trainer class (from trainer.py) handles model training loops, evaluation (accuracy, precision, recall, top-3 accuracy), and saving of training histories.
	MLP class (from classifier.py) defines an MLP network that serves as the style classifier model.
How to use:
1.	Run main.py with train_style = True (default).
2.	Adjust parameters such as hidden size and epochs within main.py if needed.
3.3.4 Phase 2-2: Training Models (Architect)
The architect classifier is trained to predict the architect based on a combination of CNN embeddings and style classifier predictions.
	train_architect_classifier() (in main.py) controls the full training pipeline for the architect classifier.
	Trainer class (from trainer.py) handles training loops, evaluation metrics, and saving results.
	MLP class (from classifier.py) defines the architect classification model, which uses concatenated embeddings (CNN features + style probabilities) as input.
How to use:
1.	Run main.py with train_architect = True (default).
2.	Adjust parameters such as hidden size and epochs within main.py if needed.
3.3.5 Result Visualization
The training performance and model evaluation results can be visualized in plotter.ipynb. Training history is automatically saved as JSON files (e.g., style_training_history.json, architect_training_history.json) during model training through the Trainer class (from trainer.py). Previous training histories can be found in:
artefact/training_history
3.3.6 GUI
A Tkinter-based desktop GUI allows users to easily upload images and receive real-time predictions.
	App class (in app.py) defines a complete graphical interface using tkinter and PIL (Pillow).
	FeatureExtractor (from feature_extractor_torch.py) is used within the app to preprocess new input images.
	The GUI requires PTH models from previous training, which have been stored inside artefact/models
How to use: 
1.	Run app.py
2.	Click "Show Background" if needed to reveal the historical and cultural context.
3.	Upload an image for prediction
4. Embeddings Description
Path: artefact/embeddings
This folder stores precomputed CNN embeddings on all tested models, which could be loaded in main.py to save time on feature extraction.  
	style_train_embeddings/: Embeddings for style classifier training set.
	style_test_embeddings/: Embeddings for style classifier test and validation set.  
	architect_train_embeddings/: Embeddings for architect classifier training set.
	architect_test_embeddings/: Embeddings for architect classifier test set.
5. Labels
Path: artefact/labels
This folder stores label files for datasets.
	style_train_labels.npy, style_test_labels.npy: Encoded style labels.
	architect_train_labels.npy, architect_test_labels.npy: Encoded architect labels.
6. Models
Path: artefact/models
The PTH models for GUI are contained in this folder, including the best performance style and architect classifier. 






