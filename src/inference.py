import os
import torch
import clip
from PIL import Image
from alert_system import alert
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
import clip
import numpy as np
from tqdm.auto import tqdm

# Load your custom classifier classes
from classifiers import CLIPB32Classifier, CLIPL14Classifier

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Label mappings
label2idx = {'person': 0, 'cat': 1, 'dog': 2, 'cow': 3, 'other': 4}
idx2label = {v: k for k, v in label2idx.items()}

class CLIPClassifierPredictor:
    def __init__(self, model_type="b32", model_dir="outputs/classifiers"):
        self.model_type = model_type.lower()
        self.model_dir = model_dir

        # Load CLIP and classifier
        if self.model_type == "b32":
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
            self.classifier = CLIPB32Classifier().to(device)
            classifier_path = os.path.join(self.model_dir, "clip_b32_classifier.pth")
        
        elif self.model_type == "l14":
            self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=device)
            self.classifier = CLIPL14Classifier().to(device)
            classifier_path = os.path.join(self.model_dir, "clip_l14_classifier.pth")
        
        else:
            raise ValueError("Invalid model_type. Choose 'b32' or 'l14'.")

        # Load classifier weights
        self.classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        self.clip_model.eval()
        self.classifier.eval()

    def predict(self, image_path):
        """Predict the class label for a single image."""
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        image = self.clip_preprocess(image).unsqueeze(0).to(device)

        # Extract CLIP features
        with torch.no_grad():
            clip_features = self.clip_model.encode_image(image).float()

            # Classify
            output = self.classifier(clip_features)
            _, pred_idx = torch.max(output, dim=1)
            pred_label = idx2label[pred_idx.item()]

        return pred_label


import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from tqdm.auto import tqdm

def plot_clip_bounding_box(
    img_path,
    label,
    clip_model,
    clip_preprocess,
    window=4,
    stride=1,
    threshold=0.8,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Plot an image with bounding box detected using CLIP and sliding window.
    Optimized for better performance.
    """

    # Load and prepare the image
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    img_tensor = transforms.ToTensor()(img)  # (C, H, W)

    patch_size = min(H, W) // 10
    patch_size = max(patch_size, 16)

    # Unfold the image into patches
    C, H_tensor, W_tensor = img_tensor.shape
    patches_tensor = img_tensor.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    n_patches_h, n_patches_w = patches_tensor.shape[1], patches_tensor.shape[2]

    # Tokenize prompt
    prompt = f"a photo of a {label}"
    text_inputs = clip.tokenize([prompt]).to(device)

    # Initialize scores and counts
    scores = torch.zeros((n_patches_h, n_patches_w), device=device)
    counts = torch.ones((n_patches_h, n_patches_w), device=device)

    # Preprocess CLIP model
    clip_model.eval()

    # Loop to process patches
    t = tqdm(total=(n_patches_h - window + 1) * (n_patches_w - window + 1), desc="Detecting Object")
    with torch.no_grad():
        for y in range(0, n_patches_h - window + 1, stride):
            for x in range(0, n_patches_w - window + 1, stride):
                # Merge small patches into a bigger patch
                big_patch = torch.zeros(3, patch_size * window, patch_size * window)

                for dy in range(window):
                    for dx in range(window):
                        big_patch[:, 
                            dy*patch_size:(dy+1)*patch_size, 
                            dx*patch_size:(dx+1)*patch_size
                        ] = patches_tensor[:, y+dy, x+dx]

                # Preprocess big patch
                big_patch_pil = transforms.ToPILImage()(big_patch)
                big_patch_input = clip_preprocess(big_patch_pil).unsqueeze(0).to(device)

                # Get CLIP features
                image_features = clip_model.encode_image(big_patch_input)
                text_features = clip_model.encode_text(text_inputs)

                # Normalize
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                # Compute similarity
                similarity = (image_features @ text_features.T).item()

                # Update scores and counts
                scores[y:y+window, x:x+window] += similarity
                counts[y:y+window, x:x+window] += 1

                t.update(1)

    # Normalize the scores
    scores /= counts
    scores = scores.cpu().numpy()

    # Postprocess scores
    scores = np.clip(scores - scores.mean(), 0, None)
    if scores.max() > 0:
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        scores = np.zeros_like(scores)

    # Detect the region above the threshold
    detection = scores > np.percentile(scores, threshold * 100)
    if detection.sum() == 0:
        print("⚠️ No object detected above threshold.")
        return

    y_indices, x_indices = np.nonzero(detection)
    y_min, y_max = y_indices.min(), y_indices.max() + 1
    x_min, x_max = x_indices.min(), x_indices.max() + 1

    # Convert patch indices to pixel coordinates
    x_min_pix = (x_min * patch_size)
    y_min_pix = (y_min * patch_size)
    width_pix = (x_max - x_min) * patch_size
    height_pix = (y_max - y_min) * patch_size

    # Scale to original image size
    scale_x = W / (W_tensor)
    scale_y = H / (H_tensor)

    x_min_pix *= scale_x
    width_pix *= scale_x
    y_min_pix *= scale_y
    height_pix *= scale_y

    # Plot image with box and label
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(img)

    rect = patches.Rectangle(
        (x_min_pix, y_min_pix),
        width_pix,
        height_pix,
        linewidth=2,
        edgecolor='red',
        facecolor='none'
    )
    ax.add_patch(rect)

    # Label background
    ax.add_patch(
        patches.Rectangle(
            (x_min_pix, y_min_pix - 11),  # shift up a bit
            len(label) * 8,  # width depending on text length
            10,
            color='red',
            alpha=0.8
        )
    )

    # Label text
    ax.text(
        x_min_pix + 2, y_min_pix - 11,  # small offset inside the red box
        label,
        color='white',
        fontsize=10,
        verticalalignment='top',
        fontweight='bold'
    )

    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Single image prediction")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image file")
    parser.add_argument("--classifier", type=str, default="b32", help="Which classifier to use: b32 or l14")
    args = parser.parse_args()

    predictor = CLIPClassifierPredictor(model_type=args.classifier)
    label = predictor.predict(args.image_path)

    alert(label)
    plot_clip_bounding_box(args.image_path, label, predictor.clip_model, predictor.clip_preprocess)
