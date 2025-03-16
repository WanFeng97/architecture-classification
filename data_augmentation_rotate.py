from PIL import Image, ImageOps, ImageEnhance
import os
import random
from collections import Counter
import shutil

def get_class_counts(dataset_path):
    """
    Returns a Counter object with the number of images in each class directory.
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
    Replace any character that is not alphanumeric or one of " ._-"
    with an underscore.
    """
    return "".join(c if c.isalnum() or c in " ._-" else "_" for c in filename)

# Augmentation functions.
def flip_horizontal(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def rotate_image_no_fill(img, angle):
    """
    Rotate the image without adding a background.
    Convert to RGBA, rotate with transparent fill, then crop to the content,
    and finally convert back to RGB.
    """
    img_rgba = img.convert("RGBA")
    rotated = img_rgba.rotate(angle, expand=True, fillcolor=(0, 0, 0, 0))
    bbox = rotated.getbbox()
    if bbox:
        rotated = rotated.crop(bbox)
    return rotated.convert("RGB")

def color_jitter(img, brightness=1.2, contrast=1.2):
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    return img

def random_augmentation(img):
    """
    Randomly apply one or more transformations.
    """
    transformations = [
        flip_horizontal,
        lambda im: rotate_image_no_fill(im, random.choice([90, 180, 270])),
        lambda im: color_jitter(im,
                                brightness=random.uniform(0.8, 1.2),
                                contrast=random.uniform(0.8, 1.2))
    ]
    aug_img = img.copy()
    for transform in random.sample(transformations, k=random.randint(1, len(transformations))):
        aug_img = transform(aug_img)
    return aug_img

def balance_augment_and_save(original_folder, balanced_folder):
    """
    Creates a balanced dataset by copying original images to balanced_folder and
    then generating augmented images for classes with fewer samples.
    
    For each class:
      - All original images are copied.
      - Then, extra images are generated (by randomly augmenting originals) until the
        total number of images for that class equals the target count (e.g. the maximum
        original class count).
    """
    os.makedirs(balanced_folder, exist_ok=True)
    
    # Get counts of original images per class.
    class_counts = get_class_counts(original_folder)
    target_count = max(class_counts.values())
    print("Original dataset counts:", class_counts)
    print("Target count per class:", target_count)
    
    for class_name in os.listdir(original_folder):
        class_dir = os.path.join(original_folder, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        # Create a folder for the class in the balanced dataset.
        balanced_class_dir = os.path.join(balanced_folder, class_name)
        os.makedirs(balanced_class_dir, exist_ok=True)
        
        # List all original image filenames.
        original_filenames = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Copy original images to the balanced folder.
        for filename in original_filenames:
            src_path = os.path.join(class_dir, filename)
            dest_path = os.path.join(balanced_class_dir, sanitize_filename(filename))
            try:
                # Option 1: Using shutil.copyfile if you simply want to copy.
                shutil.copyfile(src_path, dest_path)
            except Exception as e:
                print(f"Error copying {src_path}: {e}")
        
        current_count = len(original_filenames)
        needed = target_count - current_count
        print(f"Class: {class_name}, current: {current_count} images, need {needed} extra images.")
        
        # Generate exactly the number of extra images needed.
        for i in range(needed):
            chosen_filename = random.choice(original_filenames)
            img_path = os.path.join(class_dir, chosen_filename)
            try:
                img = Image.open(img_path)
            except Exception as e:
                print(f"Error opening {img_path}: {e}")
                continue
            aug_img = random_augmentation(img)
            base, ext = os.path.splitext(chosen_filename)
            new_filename = f"{base}_aug_bal{i}{ext}"
            new_filename = sanitize_filename(new_filename)
            save_path = os.path.join(balanced_class_dir, new_filename)
            try:
                aug_img.save(save_path)
                print(f"Saved augmented image: {save_path}")
            except Exception as e:
                print(f"Error saving {save_path}: {e}")

def print_class_counts(dataset_path):
    counts = get_class_counts(dataset_path)
    total = 0
    for class_name, count in counts.items():
        print(f"{class_name}: {count} images")
        total += count
    print(f"Total images: {total}")

# Example usage:
if __name__ == "__main__":
    original_dataset = "D:/DeepArch/arcDataset"
    balanced_dataset = "D:/DeepArch/arcDataset_balanced"
    
    balance_augment_and_save(original_dataset, balanced_dataset)
    
    print("\nClass counts in balanced dataset:")
    print_class_counts(balanced_dataset)
