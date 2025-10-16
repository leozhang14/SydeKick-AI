import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# ---- CONFIG ----
DATA_DIR    = os.path.join("sample", "segmented_data")
CSV_PATH    = os.path.join("sample", "labels.csv")
WEIGHTS     = "ufc_resnet18.pth"
PRED_CSV    = "predictions.csv"
BATCH_SIZE  = 64
NUM_WORKERS = 0

# ---- CLASSES ----
CLASSES = ["none", "punch", "kick", "grapple"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# ---- DATASET ----
class FrameDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        df = pd.read_csv(csv_path)
        df["label"] = df["label"].str.strip().str.lower()
        df["label"] = df["label"].replace({
            "none": "none",
            "punch": "punch",
            "kick": "kick",
            "grapple": "grapple"
        })

        df["img_path"] = df["frame_path"].apply(
            lambda fn: os.path.join(img_dir, os.path.splitext(fn)[0] + ".png")
        )
        df = df[df["img_path"].apply(os.path.isfile)].reset_index(drop=True)

        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["img_path"]).convert("RGB")
        label = CLASS_TO_IDX[row["label"]]
        if self.transform:
            img = self.transform(img)
        return img, label, row["frame_path"]

# ---- TRANSFORMS ----
val_tfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def main():
    dataset = FrameDataset(CSV_PATH, DATA_DIR, transform=val_tfms)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    model = model.to(device).eval()

    results = []
    correct, total = 0, 0

    with torch.no_grad():
        for x, y, paths in tqdm(loader, desc="Testing", unit="batch"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(1)

            for fp, true, pred in zip(paths, y.cpu().numpy(), preds.cpu().numpy()):
                results.append((fp, CLASSES[true], CLASSES[pred]))

            correct += (preds == y).sum().item()
            total += x.size(0)

    # Save predictions
    pred_df = pd.DataFrame(results, columns=["frame_path", "true_label", "pred_label"])
    pred_df.to_csv(PRED_CSV, index=False)
    print(f"Predictions saved to {PRED_CSV}")

    acc = correct / total
    print(f"Test accuracy: {acc*100:.2f}% ({correct}/{total} images correct)")

if __name__ == "__main__":
    main()
