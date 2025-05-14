from PIL import Image, ImageOps, ImageEnhance
import os
import random
from collections import Counter
import shutil
from sklearn.model_selection import train_test_split

# Here is designed to split the original dataset into training and test sets, all classes are preserved in both sets. 
def split_dataset(original_dataset, train_folder, test_folder, test_size=0.2):

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    
    classes = [d for d in os.listdir(original_dataset) if os.path.isdir(os.path.join(original_dataset, d))]
    
    for cls in classes:
        cls_path = os.path.join(original_dataset, cls)
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        train_imgs, test_imgs = train_test_split(images, test_size=test_size, random_state=42)
        
        train_cls_folder = os.path.join(train_folder, cls)
        test_cls_folder = os.path.join(test_folder, cls)
        os.makedirs(train_cls_folder, exist_ok=True)
        os.makedirs(test_cls_folder, exist_ok=True)
        
        for img in train_imgs:
            src = os.path.join(cls_path, img)
            dst = os.path.join(train_cls_folder, img)
            if not os.path.exists(src):
                print(f"Warning: Source file does not exist: {src}")
                continue
            try:
                shutil.copy(src, dst)
            except Exception as e:
                print(f"Error copying {src} to {dst}: {e}")
        
        for img in test_imgs:
            src = os.path.join(cls_path, img)
            dst = os.path.join(test_cls_folder, img)
            if not os.path.exists(src):
                print(f"Warning: Source file does not exist: {src}")
                continue
            try:
                shutil.copy(src, dst)
            except Exception as e:
                print(f"Error copying {src} to {dst}: {e}")

# This function is used to get the number of images in each class folder. 
def get_class_counts(dataset_path):

    class_counts = Counter()
    for class_name in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_dir):
            image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
            class_counts[class_name] = len(image_files)
    return class_counts

# Clean the filenames by replacing invalid characters with underscores.
def sanitize_filename(filename):
    return "".join(c if c.isalnum() or c in " ._-" else "_" for c in filename)

# Augmentation functions.
def flip_horizontal(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def shift_image(img, max_fraction=0.2):
    w, h = img.size
    max_shift = int(w * max_fraction)
    dx = random.randint(-max_shift, max_shift)
    if dx > 0:
        region = (dx, 0, w, h)
    elif dx < 0:
        region = (0, 0, w + dx, h)
    else:
        region = (0, 0, w, h)
    return img.crop(region)

def color_jitter(img, brightness_factor, contrast_factor):
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)
    return img

def crop_image(img, crop_fraction=0.8):
    width, height = img.size
    new_width = int(width * crop_fraction)
    new_height = int(height * crop_fraction)
    
    # Ensure we can actually crop the image
    if new_width >= width or new_height >= height:
        return img
    
    left = random.randint(0, width - new_width)
    top = random.randint(0, height - new_height)
    right = left + new_width
    bottom = top + new_height
    
    cropped_img = img.crop((left, top, right, bottom))
    # Resize back to the original dimensions
    return cropped_img.resize((width, height))

def random_augmentation(img):
    # Randomly apply one or more transformations: Horizontal flip; Left/right shift; Color jitter with pronounced effects; Random crop.
    transformations = [
        flip_horizontal,
        lambda im: shift_image(im, max_fraction=0.1),
        lambda im: color_jitter(im, 
                                brightness_factor=random.uniform(0.5, 1.5), 
                                contrast_factor=random.uniform(0.5, 1.5)),
        lambda im: crop_image(im, crop_fraction=random.uniform(0.5, 0.8))
    ]
    aug_img = img.copy()

    # Randomly choose between 1 and all available transformations.
    for transform in random.sample(transformations, k=random.randint(1, len(transformations))):
        aug_img = transform(aug_img)
    return aug_img

# Create and save the augmented dataset.
def augment_and_save_uniformly(train_folder, augmented_train_folder, augmentations_per_image=3):
    os.makedirs(augmented_train_folder, exist_ok=True)
    
    for class_name in os.listdir(train_folder):
        class_dir = os.path.join(train_folder, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        augmented_class_dir = os.path.join(augmented_train_folder, class_name)
        os.makedirs(augmented_class_dir, exist_ok=True)
        
        filenames = [f for f in os.listdir(class_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        
        for filename in filenames:
            src_path = os.path.join(class_dir, filename)
            dst_path = os.path.join(augmented_class_dir, sanitize_filename(filename))
            if not os.path.exists(src_path):
                print(f"Warning: Source file does not exist: {src_path}")
                continue
            try:
                shutil.copyfile(src_path, dst_path)
            except Exception as e:
                print(f"Error copying {src_path} to {dst_path}: {e}")
        
        for filename in filenames:
            src_path = os.path.join(class_dir, filename)
            try:
                img = Image.open(src_path)
            except Exception as e:
                print(f"Error opening {src_path}: {e}")
                continue
            for i in range(augmentations_per_image):
                aug_img = random_augmentation(img)
                base, ext = os.path.splitext(filename)
                new_filename = f"{base}_aug{i}{ext}"
                new_filename = sanitize_filename(new_filename)
                save_path = os.path.join(augmented_class_dir, new_filename)
                try:
                    aug_img.save(save_path)
                    print(f"Saved augmented image: {save_path}")
                except Exception as e:
                    print(f"Error saving {save_path}: {e}")

# print the class counts in the dataset.
def print_class_counts(dataset_path):
    counts = get_class_counts(dataset_path)
    total = sum(counts.values())
    for class_name, count in counts.items():
        print(f"{class_name}: {count} images")
    print(f"Total images: {total}")

# Main function to run the script.
if __name__ == "__main__":

    original_dataset = "../data/raw_dataset/arcDataset"
    train_folder = "../data/augmented_data/style_classification/style_train"
    test_folder = "../data/augmented_data/style_classification/style_test"
    augmented_train_folder = "../data/augmented_data/style_classification/style_train_augmented"
    
    print("Splitting dataset into training and test sets...")
    split_dataset(original_dataset, train_folder, test_folder, test_size=0.2)
    
    print("\nTrain dataset class counts:")
    print_class_counts(train_folder)
    print("\nTest dataset class counts:")
    print_class_counts(test_folder)
    
    print("\nPerforming uniform augmentation on training set...")
    augment_and_save_uniformly(train_folder, augmented_train_folder, augmentations_per_image=3)
    
    print("\nAugmented training dataset class counts:")
    print_class_counts(augmented_train_folder)

# Architect augmentation is not used in the main function, but can be run separately. Since the dataset is manually split to avoid same building in both train and test sets, it has been kept seperately. 
    '''
    test_folder = "../data/augmented_data/architect_classification/architect_11classes/architect_test"
    train_folder = "../data/augmented_data/architect_classification/architect_11classes/architect_train"
    augmented_train_folder = "../data/augmented_data/architect_classification/architect_11classes/architect_train_augmented"
    augment_and_save_uniformly(train_folder, augmented_train_folder, augmentations_per_image=3)
    
    '''

