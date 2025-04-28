import os
import requests
import zipfile
from sklearn.utils import resample
from pycocotools.coco import COCO
import pandas as pd
from tqdm import tqdm

def download_file(url, dest_path):
    """Download a file from a URL to a local destination with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    t = tqdm(total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(dest_path))

    with open(dest_path, "wb") as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()

    if total_size != 0 and t.n != total_size:
        print("ERROR: Something went wrong with the download.")

def unzip_file(zip_path, extract_to):
    """Unzip a zip file to a destination folder."""
    print(f"Extracting {os.path.basename(zip_path)}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def balance_dataset(data_dir):

    ann_file = os.path.join(data_dir, 'annotations', 'instances_val2017.json')
    images_dir = os.path.join(data_dir, 'val2017')

    # Load COCO annotations
    coco = COCO(ann_file)

    # COCO categories mapping
    coco_label_to_custom = {
        'person': 'person',
        'cat': 'cat',
        'dog': 'dog',
        'cow': 'cow'
    }

    # Priority for multi-label (lower = higher priority)
    priority = {
        'person': 0,
        'cat': 1,
        'dog': 2,
        'cow': 3,
        'other': 4
    }

    # Prepare data
    dataset_entries = []

    image_ids = coco.getImgIds()

    for img_id in tqdm(image_ids, desc="Building Custom Dataset"):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(images_dir, img_info['file_name'])

        # Get annotations
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # Collect categories in this image
        labels_in_image = []
        for ann in anns:
            cat_info = coco.loadCats(ann['category_id'])[0]
            cat_name = cat_info['name']

            if cat_name in coco_label_to_custom:
                labels_in_image.append(coco_label_to_custom[cat_name])
            else:
                labels_in_image.append('other')

        # Decide final label based on priority
        if labels_in_image:
            final_label = sorted(labels_in_image, key=lambda x: priority[x])[0]
        else:
            final_label = 'other'

        dataset_entries.append({
            'image_path': img_path,
            'label': final_label
        })

    # Save as CSV
    df = pd.DataFrame(dataset_entries)

    # Separate majority and minority classes
    majority_class = df['label'].value_counts().idxmax()
    majority_size = df['label'].value_counts().max()

    # Resample each class to match majority size
    balanced_df = pd.DataFrame()
    for label in df['label'].unique():
        class_subset = df[df['label'] == label]
        resampled = resample(class_subset,
                            replace=True,  # sample with replacement
                            n_samples=majority_size,
                            random_state=42)
        balanced_df = pd.concat([balanced_df, resampled])

    # Shuffle and save the balanced dataset
    balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)
    balanced_df.to_csv(data_dir + "/dataset.csv", index=False)

    print("\nSaved dataset!")
    print(balanced_df['label'].value_counts())


def main():
    # Define root and dataset directories
    root_dir = os.path.abspath(os.path.dirname(__file__))  # script directory
    datasets_dir = os.path.join(root_dir, "datasets")
    data_dir = os.path.join(datasets_dir, "dataset")
    
    # Create necessary directories
    os.makedirs(data_dir, exist_ok=True)

    # URLs for COCO dataset
    val_images_url = "http://images.cocodataset.org/zips/val2017.zip"
    annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

    # Paths for zip files
    val_zip_path = os.path.join(data_dir, "val2017.zip")
    ann_zip_path = os.path.join(data_dir, "annotations.zip")

    print("Downloading COCO val2017 images...")
    download_file(val_images_url, val_zip_path)

    print("\nDownloading COCO annotations...")
    download_file(annotations_url, ann_zip_path)

    print("\nExtracting COCO val2017 images...")
    unzip_file(val_zip_path, data_dir)

    print("\nExtracting COCO annotations...")
    unzip_file(ann_zip_path, data_dir)

    # Remove zip files after extraction
    os.remove(val_zip_path)
    os.remove(ann_zip_path)

    # Verify directory structure
    print("\nDownload and extraction complete!")
    print("Files inside datasets/dataset/:")
    balance_dataset(data_dir)

if __name__ == "__main__":
    main()