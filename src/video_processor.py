import os
import cv2
import torch
import clip
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

from alert_system import alert
from classifiers import CLIPB32Classifier, CLIPL14Classifier

# Label mappings
label2idx = {'person': 0, 'cat': 1, 'dog': 2, 'cow': 3, 'other': 4}
idx2label = {v: k for k, v in label2idx.items()}

device = "cuda" if torch.cuda.is_available() else "cpu"

class CLIPClassifierPredictor:
    def __init__(self, model_type="b32", model_dir="outputs/classifiers"):
        self.model_type = model_type.lower()
        self.model_dir = model_dir

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

        self.classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        self.clip_model.eval()
        self.classifier.eval()

    def predict(self, pil_image):
        """Predict the label for a PIL image directly."""
        image = self.clip_preprocess(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            clip_features = self.clip_model.encode_image(image).float()
            output = self.classifier(clip_features)
            _, pred_idx = torch.max(output, dim=1)
            pred_label = idx2label[pred_idx.item()]

        return pred_label

def detect_object(pil_img, label, clip_model, clip_preprocess, window=4, stride=1, threshold=0.8):
    img_tensor = transforms.ToTensor()(pil_img)
    W, H = pil_img.size
    C, H_tensor, W_tensor = img_tensor.shape

    patch_size = min(H, W) // 10
    patch_size = max(patch_size, 16)

    patches_tensor = img_tensor.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    n_patches_h, n_patches_w = patches_tensor.shape[1], patches_tensor.shape[2]

    prompt = f"a photo of a {label}"
    text_inputs = clip.tokenize([prompt]).to(device)

    scores = torch.zeros((n_patches_h, n_patches_w), device=device)
    counts = torch.ones((n_patches_h, n_patches_w), device=device)

    clip_model.eval()

    with torch.no_grad():
        for y in range(0, n_patches_h - window + 1, stride):
            for x in range(0, n_patches_w - window + 1, stride):
                big_patch = torch.zeros(3, patch_size * window, patch_size * window)

                for dy in range(window):
                    for dx in range(window):
                        big_patch[:, dy*patch_size:(dy+1)*patch_size, dx*patch_size:(dx+1)*patch_size] = patches_tensor[:, y+dy, x+dx]

                big_patch_pil = transforms.ToPILImage()(big_patch)
                big_patch_input = clip_preprocess(big_patch_pil).unsqueeze(0).to(device)

                image_features = clip_model.encode_image(big_patch_input)
                text_features = clip_model.encode_text(text_inputs)

                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                similarity = (image_features @ text_features.T).item()

                scores[y:y+window, x:x+window] += similarity
                counts[y:y+window, x:x+window] += 1

    scores /= counts
    scores = scores.cpu().numpy()

    scores = np.clip(scores - scores.mean(), 0, None)
    if scores.max() > 0:
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        scores = np.zeros_like(scores)

    detection = scores > np.percentile(scores, threshold * 100)
    if detection.sum() == 0:
        return None  # No detection

    y_indices, x_indices = np.nonzero(detection)
    y_min, y_max = y_indices.min(), y_indices.max() + 1
    x_min, x_max = x_indices.min(), x_indices.max() + 1

    x_min_pix = (x_min * patch_size) * (W / W_tensor)
    y_min_pix = (y_min * patch_size) * (H / H_tensor)
    width_pix = (x_max - x_min) * patch_size * (W / W_tensor)
    height_pix = (y_max - y_min) * patch_size * (H / H_tensor)

    # Pad the bounding box
    padding = 0.2
    x_min_pix = max(0, int(x_min_pix - padding * width_pix))
    y_min_pix = max(0, int(y_min_pix - padding * height_pix))
    width_pix = int(width_pix * (1 + 2 * padding))
    height_pix = int(height_pix * (1 + 2 * padding))


    return (int(x_min_pix), int(y_min_pix), int(width_pix), int(height_pix))

def process_video(video_path, output_path, predictor):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=frame_count, desc="Processing Video")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        label = predictor.predict(pil_img)

        box = detect_object(pil_img, label, predictor.clip_model, predictor.clip_preprocess)
        
        if box is not None:
            x, y, w, h = box
            thickness = max(1, int(min(frame.shape[0], frame.shape[1]) / 300))
            font_scale = min(frame.shape[0], frame.shape[1]) / 1000.0
            font_scale = max(0.5, font_scale)  # minimum readable size
            font = cv2.FONT_HERSHEY_SIMPLEX
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), thickness)

            # Draw label background
            cv2.rectangle(frame, (x, y - 20), (x + len(label) * 10, y), (0, 0, 255), -1)

            # Put label text
            cv2.putText(frame, label, (x + 2, y - 5), font, font_scale, (255, 255, 255), 1)

        out.write(frame)
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Video Detection using CLIP Classifier")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save output video")
    parser.add_argument("--classifier", type=str, default="b32", help="Which classifier to use: b32 or l14")
    args = parser.parse_args()

    predictor = CLIPClassifierPredictor(model_type=args.classifier)
    
    alert("Starting video processing!")
    process_video(args.video_path, args.output_path, predictor)
