from PIL import Image, ImageOps, ImageEnhance
import os
import random
from collections import Counter
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(original_dataset, train_folder, test_folder, test_size=0.2):
    """
    Splits the original dataset into separate training and test folders.
    Each class's images are split using train_test_split.
    """
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    
    # List all class folders in the original dataset.
    classes = [d for d in os.listdir(original_dataset) if os.path.isdir(os.path.join(original_dataset, d))]
    
    for cls in classes:
        cls_path = os.path.join(original_dataset, cls)
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        train_imgs, test_imgs = train_test_split(images, test_size=test_size, random_state=42)
        
        # Create class subfolders in train and test folders.
        train_cls_folder = os.path.join(train_folder, cls)
        test_cls_folder = os.path.join(test_folder, cls)
        os.makedirs(train_cls_folder, exist_ok=True)
        os.makedirs(test_cls_folder, exist_ok=True)
        
        # Copy training images.
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
        
        # Copy test images.
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

def get_class_counts(dataset_path):
    """
    Returns a Counter with the number of images in each class folder.
    """
    class_counts = Counter()
    for class_name in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_dir):
            image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
            class_counts[class_name] = len(image_files)
    return class_counts

def sanitize_filename(filename):
    """
    Replace any character not alphanumeric or one of " ._-" with an underscore.
    """
    return "".join(c if c.isalnum() or c in " ._-" else "_" for c in filename)

# Augmentation functions.
def flip_horizontal(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def shift_image(img, max_fraction=0.2):
    """
    Shift the image left or right by a random amount up to max_fraction of the image width.
    The function crops the image to remove any empty space.
    """
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
    """
    Apply brightness and contrast adjustments with the specified factors.
    """
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)
    return img

def random_augmentation(img):
    """
    Randomly apply one or more transformations:
      - Horizontal flip.
      - Left/right shift.
      - Color jitter with pronounced effects.
    """
    transformations = [
        flip_horizontal,
        lambda im: shift_image(im, max_fraction=0.1),
        lambda im: color_jitter(im, 
                                brightness_factor=random.uniform(0.5, 1.5), 
                                contrast_factor=random.uniform(0.5, 1.5))
    ]
    aug_img = img.copy()
    # Randomly choose between 1 and all available transformations.
    for transform in random.sample(transformations, k=random.randint(1, len(transformations))):
        aug_img = transform(aug_img)
    return aug_img

def augment_and_save_uniformly(train_folder, augmented_train_folder, augmentations_per_image=3):
    """
    Creates an augmented training dataset by copying original training images to augmented_train_folder and then 
    generating a fixed number of augmented images for each original image.
    This preserves the original distribution since each image is augmented by the same factor.
    """
    os.makedirs(augmented_train_folder, exist_ok=True)
    
    for class_name in os.listdir(train_folder):
        class_dir = os.path.join(train_folder, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        # Create a folder for the class in the augmented training set.
        augmented_class_dir = os.path.join(augmented_train_folder, class_name)
        os.makedirs(augmented_class_dir, exist_ok=True)
        
        # List all image filenames.
        filenames = [f for f in os.listdir(class_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        
        # Copy original training images.
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
        
        # For each image, generate a fixed number of augmented images.
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

def print_class_counts(dataset_path):
    """
    Prints the count of images per class in the given dataset folder.
    """
    counts = get_class_counts(dataset_path)
    total = sum(counts.values())
    for class_name, count in counts.items():
        print(f"{class_name}: {count} images")
    print(f"Total images: {total}")

if __name__ == "__main__":
    '''original_dataset = "D:/DeepArch/arcDataset_architect/Shortlist"
    train_folder = "D:/DeepArch/architect_train"
    test_folder = "D:/DeepArch/architect_test"
    augmented_train_folder = "D:/DeepArch/architect_train_augmented"
    
    print("Splitting dataset into training and test sets...")
    split_dataset(original_dataset, train_folder, test_folder, test_size=0.2)
    
    print("\nTrain dataset class counts:")
    print_class_counts(train_folder)
    print("\nTest dataset class counts:")
    print_class_counts(test_folder)
    
    print("\nPerforming uniform augmentation on training set...")
    augment_and_save_uniformly(train_folder, augmented_train_folder, augmentations_per_image=3)
    
    print("\nAugmented training dataset class counts:")
    print_class_counts(augmented_train_folder)'''

    test_folder = "D:/DeepArch/update_architect_dataset_test"
    train_folder = "D:/DeepArch/update_architect_dataset_train"
    augmented_train_folder = "D:/DeepArch/update_architect_dataset_train_augmented"

    augment_and_save_uniformly(train_folder, augmented_train_folder, augmentations_per_image=3)

