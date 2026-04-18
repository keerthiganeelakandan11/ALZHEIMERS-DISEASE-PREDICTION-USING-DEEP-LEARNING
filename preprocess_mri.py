import os
import numpy as np
import cv2
import nibabel as nib

IMG_SIZE = 128
MRI_FOLDER = "data/mri"
OUTPUT_PATH = "alzheimers_data/oasis_longitudinal.npz"

mri_images = []

print("Processing MRI folders...")

subjects = os.listdir(MRI_FOLDER)

for subject in subjects:
    raw_path = os.path.join(MRI_FOLDER, subject, "RAW")

    if not os.path.exists(raw_path):
        continue

    for file in os.listdir(raw_path):
        if file.endswith(".img"):
            img_path = os.path.join(raw_path, file)

            img = nib.load(img_path).get_fdata()

            # Take middle slice
            center = img.shape[2] // 2
            slice_img = img[:, :, center]

            # Resize
            slice_img = cv2.resize(slice_img, (IMG_SIZE, IMG_SIZE))

            # Normalize
            slice_img = slice_img / np.max(slice_img)

            mri_images.append(slice_img)

mri_images = np.array(mri_images)
mri_images = mri_images.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

print("Final MRI dataset shape:", mri_images.shape)

os.makedirs("alzheimers_data", exist_ok=True)
np.savez(OUTPUT_PATH, X_mri=mri_images)

print("Saved MRI dataset to:", OUTPUT_PATH)