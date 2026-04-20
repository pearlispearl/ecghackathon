import os
import math
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as transforms_v2

from model import ECGEfficientNet
from gradcam_utils import GradCAM, overlay_cam_on_image


# =========================
# Config
# =========================
num_classes = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# เปลี่ยน path ตามของคุณ
model_path = "model_fold1_best_f1_threshold.pth"
image_dir = "data/test_image"            # หรือ "data/ECG Signal Image"
output_dir = "gradcam_outputs"

# จะทำกี่รูป
max_images = 8

# ถ้าอยากระบุชื่อไฟล์เอง ให้ใส่ list นี้
# ถ้าไม่ใช้ ให้ปล่อยเป็น None
selected_filenames = None
# ตัวอย่าง:
# selected_filenames = [
#     "0000001_xxx.png",
#     "0000002_xxx.png",
# ]


# =========================
# Transform
# =========================
transform = transforms_v2.Compose([
    transforms_v2.ToImage(),
    transforms_v2.Resize((224, 224)),
    transforms_v2.ToDtype(torch.float32, scale=True),
    transforms_v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# Load model
# =========================
model = ECGEfficientNet(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# EfficientNet target layer
target_layer = model.backbone.features[-1]
gradcam = GradCAM(model, target_layer)


# =========================
# Prepare files
# =========================
os.makedirs(output_dir, exist_ok=True)

all_files = sorted([
    f for f in os.listdir(image_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

if selected_filenames is not None:
    image_files = [f for f in selected_filenames if f in all_files]
else:
    image_files = all_files[:max_images]

if len(image_files) == 0:
    raise ValueError("No image files found. Check image_dir or selected_filenames.")


# =========================
# Run Grad-CAM
# =========================
summary_rows = []

for fname in image_files:
    img_path = os.path.join(image_dir, fname)

    pil_img = Image.open(img_path).convert("RGB")
    pil_img_resized = pil_img.resize((224, 224))

    img_np = np.array(pil_img_resized).astype(np.uint8)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    cam, pred_class, probs = gradcam.generate(input_tensor)
    confidence = probs[0, pred_class].item()

    overlay = overlay_cam_on_image(img_np, cam, alpha=0.35)

    # save single figure
    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(img_np)
    ax1.set_title("Original ECG")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(cam, cmap="jet")
    ax2.set_title("Grad-CAM")
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(overlay)
    ax3.set_title(f"Overlay | Pred: {pred_class} | Conf: {confidence:.3f}")
    ax3.axis("off")

    plt.tight_layout()

    save_path = os.path.join(output_dir, f"gradcam_{os.path.splitext(fname)[0]}.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {save_path}")

    summary_rows.append({
        "filename": fname,
        "predicted_class": pred_class,
        "confidence": round(confidence, 6),
        "saved_path": save_path
    })


# =========================
# Save summary CSV
# =========================
summary_df = pd.DataFrame(summary_rows)
summary_csv_path = os.path.join(output_dir, "gradcam_summary.csv")
summary_df.to_csv(summary_csv_path, index=False)

print("\nDone!")
print(summary_df)
print(f"\nSummary saved to: {summary_csv_path}")

gradcam.remove_hooks()