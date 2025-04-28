import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import clip
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Argument parser
parser = argparse.ArgumentParser(description="Train a CLIP classifier")
parser.add_argument("--classifier", type=str, default="b32", help="CLIP model to use (b32 or l14)")
args = parser.parse_args()

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP and Classifier dynamically
if args.classifier == "b32":
    from src.classifiers import ClipB32Classifier
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    classifier_class = ClipB32Classifier
    model_save_name = "clip_b32_classifier.pth"

elif args.classifier == "l14":
    from src.classifiers import ClipL14Classifier
    clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
    classifier_class = ClipL14Classifier
    model_save_name = "clip_l14_classifier.pth"

else:
    raise ValueError("Invalid classifier type. Choose 'b32' or 'l14'.")

clip_model.eval()

# Dataset class
class CustomCOCODataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform
        self.label2idx = {label: idx for idx, label in enumerate(['person', 'cat', 'dog', 'cow', 'other'])}
        self.idx2label = {v: k for k, v in self.label2idx.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image_path']
        label = self.data.iloc[idx]['label']
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label_idx = self.label2idx[label]
        return image, label_idx

# Extract CLIP features for dataset
def extract_clip_features(dataset):
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Extracting features"):
            images = images.to(device)
            features = clip_model.encode_image(images).float()
            all_features.append(features.cpu())
            all_labels.append(labels)

    return torch.cat(all_features), torch.cat(all_labels)

# Training function
def train(model, criterion, optimizer, scheduler, X_train, y_train, X_val, y_val, epochs=10, patience=3):
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train.to(device))
        loss = criterion(outputs, y_train.to(device))
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val.to(device))
            val_loss = criterion(val_outputs, y_val.to(device))
            _, preds = torch.max(val_outputs, 1)
            acc = (preds.cpu() == y_val).float().mean()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Val Accuracy: {acc.item() * 100:.2f}%")

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_name)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# Evaluation function
def evaluate(model, X_test, y_test, idx2label=None):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test.to(device))
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu()

        accuracy = (preds == y_test).float().mean()
        print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")

        # Classification Report
        print("\nClassification Report:")
        target_names = [idx2label[i] for i in range(len(idx2label))] if idx2label else None
        print(classification_report(y_test, preds, target_names=target_names))

        # Confusion Matrix
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names, 
                    yticklabels=target_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

# Main script
if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("balanced_dataset.csv")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

    # In-memory split
    train_end = int(0.7 * len(df))
    val_end = int(0.85 * len(df))

    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]
    df_test.to_csv("test/test.csv", index=False)

    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711))
    ])

    val_test_transform = clip_preprocess

    # Datasets
    train_dataset = CustomCOCODataset(df_train, transform=train_transform)
    val_dataset = CustomCOCODataset(df_val, transform=val_test_transform)
    test_dataset = CustomCOCODataset(df_test, transform=val_test_transform)

    # Feature extraction
    X_train, y_train = extract_clip_features(train_dataset)
    X_val, y_val = extract_clip_features(val_dataset)
    X_test, y_test = extract_clip_features(test_dataset)

    # Model setup
    classifier = classifier_class().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)

    # Training
    train(classifier, criterion, optimizer, scheduler, X_train, y_train, X_val, y_val, epochs=200)

    # Load best model
    best_model = classifier_class().to(device)
    best_model.load_state_dict(torch.load(model_save_name))
    best_model.eval()

    # Evaluation
    evaluate(best_model, X_test, y_test, idx2label=train_dataset.idx2label)
