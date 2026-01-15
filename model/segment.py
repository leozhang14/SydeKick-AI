import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

# ---- CONFIG ----
IMAGES_DIR = os.path.join("sample", "data")
OUT_DIR    = os.path.join("sample", "segmented_data")
CSV_PATH   = os.path.join("sample", "labels.csv")

# Ensure output folder exists
os.makedirs(OUT_DIR, exist_ok=True)

# ---- DEVICE ----
device = torch.device("cuda" if torch.cuda.is_available()
                    else ("mps" if torch.backends.mps.is_available() else "cpu"))

# ---- MODEL ----
weights = DeepLabV3_ResNet101_Weights.DEFAULT
model = deeplabv3_resnet101(weights=weights).to(device).eval()

# ---- TRANSFORM ----
trf = transforms.Compose([
transforms.Resize((520, 520)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

# ---- LOAD CSV ----
df = pd.read_csv(CSV_PATH)
if "frame_path" not in df.columns or "label" not in df.columns:
raise ValueError("labels.csv must have columns: frame_path,label")

# ---- PROCESS LOOP ----
count = 0
for fname in df["frame_path"]:
in_path = os.path.join(IMAGES_DIR, fname)
if not os.path.isfile(in_path):
    continue

try:
    # Load original image
    img = Image.open(in_path).convert("RGB")
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    orig_h, orig_w = img_cv.shape[:2]

    # Transform and run model
    inp = trf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp)["out"][0]
    mask = out.argmax(0).byte().cpu().numpy()

    # Person class = 15
    person_mask = (mask == 15).astype(np.uint8) * 255
    person_mask = cv2.resize(person_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    if person_mask.max() > 0:
        # --- Keep only large blobs (fighters) ---
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(person_mask, connectivity=8)

        if num_labels > 1:
            img_area = orig_h * orig_w
            areas = stats[1:, 4]   # skip background
            comp_ids = np.arange(1, num_labels)

            # Keep only blobs >= 2% of image area
            keep = [comp_ids[i] for i, a in enumerate(areas) if a / img_area >= 0.02]

            if len(keep) == 0:
                # fallback: keep the single largest blob
                keep = [comp_ids[areas.argmax()]]

            elif len(keep) > 2:
                # if more than 2 survive, keep only the largest 2
                keep = sorted(keep, key=lambda i: stats[i, 4], reverse=True)[:2]

            mask_clean = np.zeros_like(person_mask, dtype=np.uint8)
            for comp_id in keep:
                mask_clean[labels == comp_id] = 255
            person_mask = mask_clean

    # Apply mask
    masked = cv2.bitwise_and(img_cv, img_cv, mask=person_mask)

    # Save
    base, _ = os.path.splitext(fname)
    out_path = os.path.join(OUT_DIR, f"{base}.png")
    cv2.imwrite(out_path, masked)

    count += 1
    if count % 100 == 0:
        print(f"processed {count}")

except Exception as e:
    print(f"Error on {fname}: {e}")

print(f"Done! Total processed = {count}")
