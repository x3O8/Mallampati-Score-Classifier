import os
import shutil
import random
import glob
from torchvision import transforms
from PIL import Image
from google.colab import drive

drive.mount('/content/drive')

original_base_path = '/content/drive/MyDrive/mallampati/zynn'
split_base_path = '/content/drive/MyDrive/mallampati/split_data'
augmented_base_path = '/content/drive/MyDrive/mallampati/augmented_data'
classes = ['1', '2', '3', '4']

print("--- Starting Dataset Split ---")
if os.path.exists(split_base_path):
    shutil.rmtree(split_base_path)

for split in ['train', 'val', 'test']:
    for c in classes:
        os.makedirs(os.path.join(split_base_path, split, c), exist_ok=True)

val_split = 0.15
test_split = 0.15

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

    for img in train_imgs:
        shutil.copy(os.path.join(class_path, img), os.path.join(split_base_path, 'train', c, img))
    for img in val_imgs:
        shutil.copy(os.path.join(class_path, img), os.path.join(split_base_path, 'val', c, img))
    for img in test_imgs:
        shutil.copy(os.path.join(class_path, img), os.path.join(split_base_path, 'test', c, img))

    print(f"Class '{c}': {num_train} train, {num_val} val, {num_test} test")

print("\n--- Dataset Splitting Complete ---")

print("\n--- Starting Data Augmentation ---")

augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=8),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1))
])

source_train_path = os.path.join(split_base_path, 'train')
augmented_train_path = os.path.join(augmented_base_path, 'train')

if os.path.exists(augmented_base_path):
    shutil.rmtree(augmented_base_path)

os.makedirs(augmented_train_path, exist_ok=True)

shutil.copytree(os.path.join(split_base_path, 'val'), os.path.join(augmented_base_path, 'val'))
shutil.copytree(os.path.join(split_base_path, 'test'), os.path.join(augmented_base_path, 'test'))

for c in classes:
    class_path = os.path.join(source_train_path, c)
    dest_path = os.path.join(augmented_train_path, c)
    os.makedirs(dest_path, exist_ok=True)

    images = glob.glob(os.path.join(class_path, '*'))
    num_original_images = len(images)

    target_count = 150
    augment_factor = (target_count // num_original_images)

    for img_path in images:
        shutil.copy(img_path, dest_path)
        base_name = os.path.basename(img_path)
        for i in range(augment_factor):
            original_image = Image.open(img_path).convert("RGB")
            augmented_image = augmentation_transforms(original_image)
            aug_filename = f"aug_{i}_{base_name}"
            augmented_image.save(os.path.join(dest_path, aug_filename))

    print(f"Augmented Class '{c}': Original={num_original_images}, New Total≈{len(os.listdir(dest_path))}")

print("\n--- Data Augmentation Complete ---")
print(f"Your final, ready-to-use dataset is located at: {augmented_base_path}")
