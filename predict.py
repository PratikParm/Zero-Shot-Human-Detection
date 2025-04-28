import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import clip
import matplotlib.pyplot as plt
from src.classifiers import CLIPB32Classifier
import numpy as np

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Predict function
def predict_images(folder_path, classifier, clip_model, clip_preprocess, idx2label):
    results = []
    images = []

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    for img_name in image_files:
        img_path = os.path.join(folder_path, img_name)
        image = Image.open(img_path).convert("RGB")
        images.append(image)

        input_img = clip_preprocess(image).unsqueeze(0).to(device)  # Add batch dimension

        with torch.no_grad():
            features = clip_model.encode_image(input_img).float()
            output = classifier(features)
            _, pred = output.max(1)
            label = idx2label[pred.item()]

        results.append((img_name, label))

    return images, results

# Display grid
def plot_predictions(images, predictions, grid_size=(4, 4), patch_size=256, window=6, stride=1, threshold=0.5):
    num_images = len(images)
    rows, cols = grid_size

    plt.figure(figsize=(16, 12))

    for idx in range(min(num_images, rows * cols)):
        img = images[idx]
        label = predictions[idx][1]  # Assuming (predicted_index, predicted_label)

        # Convert to tensor if needed
        if isinstance(img, np.ndarray):
            img_tensor = torch.from_numpy(np.moveaxis(img, -1, 0)).float()
        else:
            img_tensor = transforms.ToTensor()(img)

        # Detect bounding box using CLIP sliding window
        img_patches = get_patches(img_tensor, patch_size=patch_size)
        scores = get_scores(img_patches, f"a photo of a {label}", window=window, stride=stride)
        x, y, width, height = get_box(scores, patch_size=patch_size, threshold=threshold)

        # Plot image
        plt.subplot(rows, cols, idx + 1)
        image_np = np.moveaxis(img_tensor.numpy(), 0, -1)
        plt.imshow(image_np)

        # Draw bounding box
        ax = plt.gca()
        rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

        plt.title(label, fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.show()



import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load CLIP model and processor once
clip_model_id = "openai/clip-vit-large-patch14"
clip_model = CLIPModel.from_pretrained(clip_model_id).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_model_id)

# ToTensor transform
transt = transforms.ToTensor()

def get_patches(img, patch_size):
    patches = img.unfold(0, 3, 3)  # color channels
    patches = patches.unfold(1, patch_size, patch_size)
    patches = patches.unfold(2, patch_size, patch_size)
    return patches

def get_scores(img_patches, prompt, window, stride):
    patch = img_patches.shape[-1]
    scores = torch.zeros(img_patches.shape[1], img_patches.shape[2])
    runs = torch.ones(img_patches.shape[1], img_patches.shape[2])

    for Y in range(0, img_patches.shape[1] - window + 1, stride):
        for X in range(0, img_patches.shape[2] - window + 1, stride):
            big_patch = torch.zeros(patch*window, patch*window, 3)
            patch_batch = img_patches[0, Y:Y+window, X:X+window]
            for y in range(window):
                for x in range(window):
                    big_patch[
                        y*patch:(y+1)*patch, x*patch:(x+1)*patch, :
                    ] = patch_batch[y, x].permute(1, 2, 0)
            inputs = clip_processor(
                images=big_patch,
                text=prompt,
                return_tensors="pt",
                padding=True
            ).to(device)
            score = clip_model(**inputs).logits_per_image.item()
            scores[Y:Y+window, X:X+window] += score
            runs[Y:Y+window, X:X+window] += 1

    scores /= runs
    scores = np.clip(scores - scores.mean(), 0, np.inf)
    scores = (scores - scores.min()) / (scores.max() - scores.min())
    return scores

def get_box(scores, patch_size, threshold):
    detection = scores > threshold
    if detection.sum() == 0:
        # No strong match, return full image
        return 0, 0, patch_size, patch_size

    y_min, y_max = (
        np.nonzero(detection)[:,0].min().item(),
        np.nonzero(detection)[:,0].max().item()+1
    )
    x_min, x_max = (
        np.nonzero(detection)[:,1].min().item(),
        np.nonzero(detection)[:,1].max().item()+1
    )

    y_min *= patch_size
    y_max *= patch_size
    x_min *= patch_size
    x_max *= patch_size

    width = x_max - x_min
    height = y_max - y_min
    return x_min, y_min, width, height

def detect_and_draw(img_pil, predicted_label, patch_size=256, window=6, stride=1, threshold=0.5):
    img_tensor = transt(img_pil)
    img_patches = get_patches(img_tensor, patch_size)
    scores = get_scores(img_patches, f"a photo of a {predicted_label}", window, stride)
    x, y, width, height = get_box(scores, patch_size, threshold)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    image = np.moveaxis(img_tensor.numpy(), 0, -1)
    ax.imshow(image)
    rect = patches.Rectangle((x, y), width, height, linewidth=3, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    plt.title(f"Predicted: {predicted_label}")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True, help="Path to folder containing test images")
    parser.add_argument("--classifier", type=str, default="b32", help="CLIP model to use (b32 or l14)")
    args = parser.parse_args()

    model_type = args.classifier.lower()
    model_dir = 'outputs/classifiers'

    # Load CLIP
    if model_type == "b32":
        from src.classifiers import CLIPB32Classifier
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        classifier = CLIPB32Classifier().to(device)
        classifier.load_state_dict(torch.load(os.path.join(model_dir, "clip_b32_classifier.pth")))

    elif model_type == "l14":
        from src.classifiers import CLIPL14Classifier
        clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
        classifier = CLIPL14Classifier().to(device)
        classifier.load_state_dict(torch.load(os.path.join(model_dir, "clip_l14_classifier.pth")))
    
    clip_model.eval()
    classifier.eval()
    for param in clip_model.parameters():
        param.requires_grad = False

    # Load trained classifier

    # Label mappings
    label2idx = {'person': 0, 'cat': 1, 'dog': 2, 'cow': 3, 'other': 4}
    idx2label = {v: k for k, v in label2idx.items()}

    images, predictions = predict_images(args.folder, 
                                         classifier, 
                                         clip_model, 
                                         clip_preprocess,
                                         idx2label)


    # Plot predictions
    plot_predictions(images, predictions, grid_size=(4, 4))  # 4x4 grid
