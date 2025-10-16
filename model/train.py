import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms

# ---- CONFIG ----
DATA_DIR    = os.path.join("sample", "segmented_data")
CSV_PATH    = os.path.join("sample", "labels.csv")
OUT_WEIGHTS = "ufc_resnet18.pth"
BATCH_SIZE  = 64
EPOCHS      = 10
LR          = 1e-3
NUM_WORKERS = 0   # safer for macOS/Windows
TRAIN_RATIO = 0.8

# ---- CLASSES ----
CLASSES = ["none", "punch", "kick", "grapple"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# ---- DATASET ----
class FrameDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        df = pd.read_csv(csv_path)
        if "frame_path" not in df.columns or "label" not in df.columns:
            raise ValueError("CSV must have columns: frame_path,label")

        # Map labels to known classes
        df["label"] = df["label"].str.strip().str.lower()
        df["label"] = df["label"].replace({
            "none": "none",
            "punch": "punch",
            "kick": "kick",
            "grapple": "grapple"
        })

        # Attach image path
        df["img_path"] = df["frame_path"].apply(
            lambda fn: os.path.join(img_dir, os.path.splitext(fn)[0] + ".png")
        )
        df = df[df["img_path"].apply(os.path.isfile)].reset_index(drop=True)

        bad = df[~df["label"].isin(CLASSES)]
        if len(bad) > 0:
            raise ValueError(f"Found unknown labels: {bad['label'].unique()}")

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
        return img, label

# ---- TRANSFORMS ----
train_tfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

val_tfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def main():
    # Dataset + split
    dataset = FrameDataset(CSV_PATH, DATA_DIR, transform=None)
    n_total = len(dataset)
    n_train = int(n_total * TRAIN_RATIO)
    n_val   = n_total - n_train

    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_ds.dataset.transform = train_tfms
    val_ds.dataset.transform   = val_tfms

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"Train size: {n_train} | Validation size: {n_val}")

    # Model
    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model = model.to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Training loop
    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss, correct, total = 0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)

            pbar.set_postfix(loss=loss.item(), acc=correct/total)

        train_loss = total_loss / total
        train_acc = correct / total
        print(f"Epoch {epoch}: Train loss {train_loss:.4f}, Train acc {train_acc:.4f}")

    # Save weights
    torch.save(model.state_dict(), OUT_WEIGHTS)
    print(f"Model saved to {OUT_WEIGHTS}")

if __name__ == "__main__":
    main()
