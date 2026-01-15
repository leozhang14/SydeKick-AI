import os
import sys
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import pandas as pd
from collections import deque
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

# -------- CONFIG --------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "ufc_resnet18.pth")
CLASSES = ["none", "punch", "kick", "grapple"]
FPS = 30.0
CONFIDENCE_THRESHOLD = 0.90
FRAME_BUFFER_SIZE = 5
ACTION_CLASSES = ["punch", "kick", "grapple"]  # Actions to track (exclude "none")

device = torch.device("cuda" if torch.cuda.is_available() 
                     else ("mps" if torch.backends.mps.is_available() else "cpu"))

# -------- SEGMENTATION MODEL --------
print("Loading segmentation model...")
seg_weights = DeepLabV3_ResNet101_Weights.DEFAULT
seg_model = deeplabv3_resnet101(weights=seg_weights).eval().to(device)

seg_transform = transforms.Compose([
    transforms.Resize((520, 520)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def segment_frame(frame):
    """Segment frame to keep only person blobs (fighters)"""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    orig_h, orig_w = frame.shape[:2]
    
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
    masked = cv2.bitwise_and(frame, frame, mask=person_mask)
    return masked

# -------- CLASSIFICATION MODEL --------
print("Loading classification model...")
clf_model = models.resnet18(weights=None)
num_ftrs = clf_model.fc.in_features
clf_model.fc = nn.Linear(num_ftrs, len(CLASSES))
clf_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
clf_model.eval().to(device)

clf_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def classify_frame(frame):
    """Classify a segmented frame"""
    # Convert BGR to RGB for PIL
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inp = clf_transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = clf_model(inp)
        probs = torch.softmax(outputs, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)
    
    pred_class = CLASSES[pred_idx.item()]
    confidence = conf.item()
    
    return pred_class, confidence

# -------- VIDEO PROCESSING FUNCTION --------
def process_video(video_path):
    """
    Process MP4 video file and detect actions
    video_path: str (path to video file)
    """
    if not os.path.exists(video_path):
        raise ValueError(f"Video file not found: {video_path}")
    
    # Extract base name without extension for CSV
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    csv_filename = f"{base_name}.csv"
    csv_path = os.path.join(SCRIPT_DIR, csv_filename)
    
    print(f"Processing video file: {video_path}")
    print(f"Output CSV will be saved to: {csv_path}")
    
    # Initialize CSV file
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        print(f"Found existing CSV with {len(df_existing)} entries. Appending new detections...")
    else:
        df_existing = pd.DataFrame(columns=["timestamp", "action", "confidence"])
        # Create empty CSV with headers
        df_existing.to_csv(csv_path, index=False)
    
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")
    
    # Get actual FPS from video
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps > 0:
        fps = actual_fps
        print(f"Detected FPS: {fps}")
    else:
        fps = FPS
        print(f"Could not detect FPS, using default: {fps}")
    
    # Get total frame count for progress tracking
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")
    
    # Frame buffer to track last N predictions
    frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)
    frame_count = 0
    last_recorded_frame = -FRAME_BUFFER_SIZE  # Prevent duplicate recordings
    
    print("Processing frames...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Progress update
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"Processing frame {frame_count}/{total_frames} ({progress:.1f}%)")
            
            # Segment frame
            segmented = segment_frame(frame)
            
            # Classify frame
            pred_class, confidence = classify_frame(segmented)
            
            # Add to buffer
            frame_buffer.append((pred_class, confidence, frame_count))
            
            # Check if we have enough frames in buffer
            if len(frame_buffer) == FRAME_BUFFER_SIZE:
                # Check if all last 5 frames have same action, confidence > threshold, and action is in ACTION_CLASSES
                classes_in_buffer = [item[0] for item in frame_buffer]
                confidences_in_buffer = [item[1] for item in frame_buffer]
                
                # All same class, all confidences > threshold, and class is an action (not "none")
                if (len(set(classes_in_buffer)) == 1 and 
                    classes_in_buffer[0] in ACTION_CLASSES and
                    all(c > CONFIDENCE_THRESHOLD for c in confidences_in_buffer) and
                    frame_count - last_recorded_frame >= FRAME_BUFFER_SIZE):
                    
                    # Calculate timestamp
                    timestamp = frame_count / fps
                    action = classes_in_buffer[0]
                    avg_confidence = np.mean(confidences_in_buffer)
                    
                    # Record to CSV
                    new_row = pd.DataFrame({
                        "timestamp": [timestamp],
                        "action": [action],
                        "confidence": [avg_confidence]
                    })
                    df_existing = pd.concat([df_existing, new_row], ignore_index=True)
                    df_existing.to_csv(csv_path, index=False)
                    
                    last_recorded_frame = frame_count
                    print(f"Frame {frame_count}: Detected {action} at {timestamp:.2f}s (confidence: {avg_confidence:.4f})")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        # Ensure CSV is saved at the end
        df_existing.to_csv(csv_path, index=False)
        print(f"\nProcessing complete. Results saved to {csv_path}")
        print(f"Total frames processed: {frame_count}")
        print(f"Total detections recorded: {len(df_existing)}")

# -------- MAIN --------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python upload.py <video_file.mp4>")
        print("Example: python upload.py fight.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    process_video(video_path)
