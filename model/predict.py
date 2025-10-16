import os
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

# -------- CONFIG --------
MODEL_PATH = "ufc_resnet18.pth"   # fine-tuned weights
IMG_PATH   = "sample_kick.png"   # input raw fight image
CLASSES    = ["none", "punch", "kick", "grapple"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- SEGMENTATION MODEL --------
weights = DeepLabV3_ResNet101_Weights.DEFAULT
seg_model = deeplabv3_resnet101(weights=weights).eval().to(device)

seg_transform = transforms.Compose([
    transforms.Resize((520, 520)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def segment_image(img_path):
    """Remove background, keep 2 largest person blobs"""
    img = Image.open(img_path).convert("RGB")
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    orig_h, orig_w = img_cv.shape[:2]

    inp = seg_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = seg_model(inp)["out"][0]
    mask = out.argmax(0).byte().cpu().numpy()

    # Person = 15 in COCO
    person_mask = (mask == 15).astype(np.uint8) * 255
    person_mask = cv2.resize(person_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Keep only 2 largest blobs
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(person_mask, connectivity=8)
    if num_labels > 1:
        areas = stats[1:, 4]
        largest_ids = areas.argsort()[-2:] + 1
        mask_clean = np.zeros_like(person_mask)
        for comp_id in largest_ids:
            mask_clean[labels == comp_id] = 255
        person_mask = mask_clean

    # Apply mask
    masked = cv2.bitwise_and(img_cv, img_cv, mask=person_mask)
    segmented_path = "segmented_input.png"
    cv2.imwrite(segmented_path, masked)
    return segmented_path

# -------- CLASSIFICATION MODEL --------
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval().to(device)

clf_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# -------- RUN PIPELINE --------
segmented_path = segment_image(IMG_PATH)

img = Image.open(segmented_path).convert("RGB")
inp = clf_transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(inp)
    probs = torch.softmax(outputs, dim=1)[0]
    conf, pred_idx = torch.max(probs, dim=0)

pred_class = CLASSES[pred_idx.item()]
print(f"Prediction: {pred_class} (confidence {conf.item():.4f})")

# Save confidences
with open("prediction_confidences.csv", "w") as f:
    f.write("class,confidence\n")
    for i, cls in enumerate(CLASSES):
        f.write(f"{cls},{probs[i].item():.4f}\n")

print("Saved per-class confidences to prediction_confidences.csv")
