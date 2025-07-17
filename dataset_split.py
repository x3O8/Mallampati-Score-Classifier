import os
import shutil
import random
import glob
from torchvision import transforms
from PIL import Image

# Mount Google Drive (if you haven't already)
from google.colab import drive
drive.mount('/content/drive')

# --- 1. SETUP PATHS ---
# Path to your original, unsplit dataset
original_base_path = '/content/drive/MyDrive/MPC_Img/Dataset'

# Path where the new split dataset (train/val/test) will be created
split_base_path = '/content/drive/MyDrive/mallampati/split_data'

# Path where the final augmented dataset for training will be stored
augmented_base_path = '/content/drive/MyDrive/mallampati/augmented_data'

classes = ['1', '2', '3', '4']

# --- 2. SPLIT THE ORIGINAL DATASET ---
print("--- Starting Dataset Split ---")
# Clear existing directories if they exist to ensure a fresh start
if os.path.exists(split_base_path):
    shutil.rmtree(split_base_path)

# Create train, validation, and test directories
for split in ['train', 'val', 'test']:
    for c in classes:
        os.makedirs(os.path.join(split_base_path, split, c), exist_ok=True)

# Define split ratios
val_split = 0.1
test_split = 0.1

for c in classes:
    class_path = os.path.join(original_base_path, c)
    images = os.listdir(class_path)
    random.shuffle(images)

    num_images = len(images)
    num_val = int(num_images * val_split)
    num_test = int(num_images * test_split)
    num_train = num_images - num_val - num_test

    train_imgs = images[:num_train]
    val_imgs = images[num_train : num_train + num_val]
    test_imgs = images[num_train + num_val :]

    # Copy files to new directories
    for img in train_imgs:
        shutil.copy(os.path.join(class_path, img), os.path.join(split_base_path, 'train', c, img))
    for img in val_imgs:
        shutil.copy(os.path.join(class_path, img), os.path.join(split_base_path, 'val', c, img))
    for img in test_imgs:
        shutil.copy(os.path.join(class_path, img), os.path.join(split_base_path, 'test', c, img))

    print(f"Class '{c}': {num_train} train, {num_val} val, {num_test} test")

print("\n--- Dataset Splitting Complete ---")


# --- 3. AUGMENT THE TRAINING DATA ---
print("\n--- Starting Data Augmentation ---")

# Define medically-appropriate augmentations
# These are slight variations to simulate real-world conditions
augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=8),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1))
])

# Path to the split training data
source_train_path = os.path.join(split_base_path, 'train')

# Path for the new, augmented training data
augmented_train_path = os.path.join(augmented_base_path, 'train')

# Clear existing augmented data directory
if os.path.exists(augmented_base_path):
    shutil.rmtree(augmented_base_path)

# Create the new augmented data structure
os.makedirs(augmented_train_path, exist_ok=True)

# Copy the validation and test sets without augmentation
shutil.copytree(os.path.join(split_base_path, 'val'), os.path.join(augmented_base_path, 'val'))
shutil.copytree(os.path.join(split_base_path, 'test'), os.path.join(augmented_base_path, 'test'))

# Augment the training data
# We aim for ~150-200 images per class to provide a decent training set
for c in classes:
    class_path = os.path.join(source_train_path, c)
    dest_path = os.path.join(augmented_train_path, c)
    os.makedirs(dest_path, exist_ok=True)

    images = glob.glob(os.path.join(class_path, '*'))
    num_original_images = len(images)

    # Calculate how many augmented versions to create per image
    # Aim for a target of around 150 images in each class folder
    target_count = 150
    augment_factor = (target_count // num_original_images)

    for img_path in images:
        # 1. Copy the original image
        shutil.copy(img_path, dest_path)

        # 2. Create augmented versions
        base_name = os.path.basename(img_path)
        for i in range(augment_factor):
            original_image = Image.open(img_path).convert("RGB")
            augmented_image = augmentation_transforms(original_image)
            aug_filename = f"aug_{i}_{base_name}"
            augmented_image.save(os.path.join(dest_path, aug_filename))

    print(f"Augmented Class '{c}': Original={num_original_images}, New Total≈{len(os.listdir(dest_path))}")

print("\n--- Data Augmentation Complete ---")
print(f"Your final, ready-to-use dataset is located at: {augmented_base_path}")