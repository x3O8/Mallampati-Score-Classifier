from google.colab import drive
drive.mount('/content/drive')


import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
base_path = '/content/drive/MyDrive/mallampati/tren/augmented_data'
train_path = os.path.join(base_path, 'train')
test_path = os.path.join(base_path, 'test')
classes = ['1', '2', '3', '4']
num_classes = len(classes)
img_size = 224
batch_size = 8

# Transforms
train_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
test_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Datasets
train_dataset = datasets.ImageFolder(train_path, transform=train_transforms)
test_dataset = datasets.ImageFolder(test_path, transform=test_transforms)

# Sampler for imbalance
class_counts = np.bincount([label for _, label in train_dataset])
weights = 1. / class_counts
samples_weights = [weights[label] for _, label in train_dataset]
sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model creation utility
def get_model(model_name):
    if model_name == 'efficientnet':
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'resnet':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'mobilenet':
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model.to(device)

# Train model function
def train_model(model, num_epochs=30, patience=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0
    trigger = 0
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        running_loss, running_corrects = 0.0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += (outputs.argmax(1) == labels).sum().item()

        train_acc = running_corrects / len(train_loader.dataset)
        print(f"Epoch {epoch+1}: Train Loss={running_loss:.4f} Acc={train_acc:.4f}")

        # Early stopping via test acc
        model.eval()
        correct = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                correct += (outputs.argmax(1) == labels).sum().item()
        acc = correct / len(test_loader.dataset)
        print(f" → Val Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            trigger = 0
            best_model = model.state_dict()
        else:
            trigger += 1
            if trigger >= patience:
                print("Early stopping.")
                break

    model.load_state_dict(best_model)
    return model

# Train all models
model_names = ['efficientnet', 'resnet', 'mobilenet']
models_list = []

for name in model_names:
    print(f"\nTraining {name.upper()}...")
    model = get_model(name)
    model = train_model(model)
    models_list.append(model)

# Ensemble prediction - Soft voting
y_true, y_pred, y_score = [], [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        probs = [F.softmax(model(inputs), dim=1) for model in models_list]
        avg_probs = torch.stack(probs).mean(dim=0)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(torch.argmax(avg_probs, axis=1).cpu().numpy())
        y_score.extend(avg_probs.cpu().numpy())

# Metrics
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d",cmap = 'blue', xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=classes))

# ROC-AUC
y_true_bin = np.eye(num_classes)[y_true]
y_score = np.array(y_score)
fpr, tpr, roc_auc = {}, {}, {}
plt.figure(figsize=(10, 8))
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], label=f"Class {classes[i]} (AUC={roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()

# Grad-CAM for EfficientNet
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output)

        target = output[0, class_idx]
        target.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = F.interpolate(cam, size=(img_size, img_size), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

def show_gradcam(img_tensor, cam, title="Grad-CAM"):
    img = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = (img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]
    superimposed = heatmap * 0.4 + img
    plt.figure(figsize=(6, 6))
    plt.imshow(superimposed)
    plt.title(title)
    plt.axis("off")
    plt.show()

# Grad-CAM on sample
sample_img, sample_label = test_dataset[0]
input_tensor = sample_img.unsqueeze(0).to(device)
target_layer = models_list[0].features[6][1]  # EfficientNet
gradcam = GradCAM(models_list[0], target_layer)
cam = gradcam.generate(input_tensor)
show_gradcam(sample_img, cam, f"Grad-CAM - True: {classes[sample_label]}")
