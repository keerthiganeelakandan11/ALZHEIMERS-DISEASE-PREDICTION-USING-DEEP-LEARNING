import numpy as np
import cv2
import tensorflow as tf
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # ✅ ADDED
import joblib
import pickle                                                          # ✅ ADDED
import os
import streamlit as st

# -------------------------------
# LOAD MODEL
# -------------------------------
model = tf.keras.models.load_model("alzheimers_model.keras")

IMG_SIZE = 160

# -------------------------------
# LOAD DATA
# -------------------------------
clinical_df = pd.read_excel("oasis_longitudinal_demographics.xlsx")

# ✅ FIX 1 — correct scaler filename (train.py saves as "clinical_scaler.pkl")
if os.path.exists("clinical_scaler.pkl"):
    scaler = joblib.load("clinical_scaler.pkl")
elif os.path.exists("scaler.pkl"):
    scaler = joblib.load("scaler.pkl")
else:
    scaler = StandardScaler()  # fallback (not ideal)

# -------------------------------
# CLASS LABELS
# -------------------------------
stage_mapping = {
    0: "Non Demented",
    1: "Very Mild Dementia",
    2: "Moderate Dementia"
}

# -------------------------------
# MRI PREPROCESSING
# -------------------------------
def preprocess_mri(img_path):
    img = nib.load(img_path).get_fdata()
    depth = img.shape[2]
    center = depth // 2

    slices = []

    for i in range(max(0, center - 12), min(depth, center + 13)):
        slice_img = img[:, :, i]

        slice_img = cv2.resize(slice_img, (IMG_SIZE, IMG_SIZE)).astype(np.float32)

        # normalize
        slice_img = (slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img) + 1e-8)

        slice_img = np.expand_dims(slice_img, -1)
        slices.append(slice_img)

    return np.array(slices)

# -------------------------------
# UI START
# -------------------------------
st.set_page_config(layout="wide")
st.title("🧠 Alzheimer Disease Prediction System")
st.markdown("---")

# -------------------------------
# MRI UPLOAD
# -------------------------------
st.header("Upload MRI Scan (.img)")

uploaded_file = st.file_uploader("Upload MRI (.img file)", type=["img", "nii"])

mri_slices = None
preview = None

if uploaded_file is not None:
    with open("temp_mri.img", "wb") as f:
        f.write(uploaded_file.read())

    mri_slices = preprocess_mri("temp_mri.img")
    preview = mri_slices[len(mri_slices) // 2].squeeze()

# -------------------------------
# LAYOUT
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("MRI Brain Scan")
    if preview is not None:
        st.image(preview, width=300)
    else:
        st.warning("Upload MRI to view image")

with col2:
    st.subheader("Clinical Data")

    subject_id = st.text_input("Enter Subject ID (Example: OAS2_0102)")

    clinical_input = None

    if subject_id:
        row = clinical_df[clinical_df["Subject ID"] == subject_id]

        if len(row) > 0:
            row = row.iloc[0]

            age = row["Age"]
            mmse = row["MMSE"]
            ses = row["SES"]
            etiv = row["eTIV"]
            nwbv = row["nWBV"]
            asf = row["ASF"]
            gender = 1 if row["M/F"] == "M" else 0

            # DISPLAY ALL 7 FEATURES
            st.write("Age:", age)
            st.write("MMSE:", mmse)
            st.write("SES:", ses)
            st.write("nWBV:", nwbv)
            st.write("eTIV:", etiv)
            st.write("ASF:", asf)
            st.write("Gender:", "Male" if gender == 1 else "Female")

            # ✅ FIX 2 — feature order must match train.py exactly:
            # train.py: [age, mmse, etiv, nwbv, asf, ses, gender]
            clinical_features = np.array([[age, mmse, etiv, nwbv, asf, ses, gender]], dtype=np.float32)

            # SCALE INPUT
            try:
                clinical_input = scaler.transform(clinical_features)
            except:
                clinical_input = scaler.fit_transform(clinical_features)

        else:
            st.error("Clinical data not found")

# -------------------------------
# PREDICTION
# -------------------------------
st.markdown("---")

if st.button("Predict Alzheimer Stage"):

    if mri_slices is None:
        st.error("Please upload MRI scan")
    elif clinical_input is None:
        st.error("Please enter valid Subject ID")
    else:
        predictions = []
        probs = []

        for slice_img in mri_slices:
            mri_input = np.expand_dims(slice_img, axis=0)

            pred = model.predict([mri_input, clinical_input], verbose=0)

            predictions.append(np.argmax(pred))
            probs.append(pred)

        predictions = np.array(predictions)
        probs = np.array(probs)

        final_pred = np.bincount(predictions).argmax()
        avg_pred = np.mean(probs, axis=0)

        stage = stage_mapping[final_pred]
        confidence = avg_pred[0][final_pred] * 100

        # -------------------------------
        # ✅ MODIFICATION 1 — PREDICTION + CONFIDENCE (color-coded)
        # -------------------------------
        st.header("Prediction Results")

        if stage == "Non Demented":
            st.success(f"✅ Prediction: **{stage}**")
        elif stage == "Very Mild Dementia":
            st.warning(f"⚠️ Prediction: **{stage}**")
        else:
            st.error(f"🚨 Prediction: **{stage}**")

        st.write(f"**Confidence: {confidence:.2f}%**")
        st.progress(int(confidence))

        # -------------------------------
        # ✅ MODIFICATION 2 — SUMMARY TABLE (all clinical fields)
        # -------------------------------
        st.subheader("Summary Table")

        summary = pd.DataFrame({
            "Feature": ["Prediction", "Confidence", "Age", "MMSE", "SES", "nWBV", "eTIV", "ASF", "Gender"],
            "Value": [
                str(stage),
                f"{confidence:.2f}%",
                str(age), str(mmse), str(ses), str(nwbv), str(etiv), str(asf),
                "Male" if gender == 1 else "Female"
            ]
        })

        st.table(summary)

        # -------------------------------
        # Class Probabilities (unchanged)
        # -------------------------------
        st.subheader("Class Probabilities")

        prob_df = pd.DataFrame(avg_pred, columns=list(stage_mapping.values()))

        # -------------------------------
        # ✅ MODIFICATION 3 — GRAPH (colors + value labels on bars)
        # -------------------------------
        labels = list(stage_mapping.values())

        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ['#2ecc71' if l == stage else '#4A90D9' for l in labels]
        bars = ax.bar(labels, avg_pred[0][:len(labels)], color=colors, edgecolor='white')
        for bar, val in zip(bars, avg_pred[0][:len(labels)]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{val:.2%}', ha='center', va='bottom', fontsize=10)
        ax.set_title("Alzheimer Stage Prediction Probabilities")
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=20)
        plt.tight_layout()
        st.pyplot(fig)

        # -------------------------------
        # ✅ MODIFICATION 4 — CONFUSION MATRIX
        # -------------------------------
        st.subheader("🔲 Confusion Matrix (Model Evaluation)")
        try:
            X_test_mri      = np.load("X_test_mri.npy")
            X_test_clinical = np.load("X_test_clinical.npy")
            y_test          = np.load("y_test.npy")
            y_pred          = model.predict([X_test_mri, X_test_clinical], verbose=0)
            y_pred_classes  = np.argmax(y_pred, axis=1)
            cm   = confusion_matrix(y_test, y_pred_classes)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
            disp.plot(ax=ax_cm, colorbar=True, cmap='Blues')
            ax_cm.set_title("Confusion Matrix", fontweight='bold')
            plt.xticks(rotation=15, ha='right')
            plt.tight_layout()
            st.pyplot(fig_cm)
        except:
            st.info("ℹ️ Confusion matrix available only during model evaluation. Save X_test_mri.npy, X_test_clinical.npy, y_test.npy")

        # -------------------------------
        # ✅ MODIFICATION 5 — ACCURACY & LOSS PLOTS
        # -------------------------------
        st.subheader("📉 Training Performance")
        try:
            with open("history.pkl", "rb") as f:
                hist = pickle.load(f)

            fig1, ax1 = plt.subplots(figsize=(7, 4))
            ax1.plot(hist['accuracy'],     label='Train Accuracy',      color='#4A90D9', linewidth=2)
            ax1.plot(hist['val_accuracy'], label='Validation Accuracy', color='#E74C3C', linewidth=2)
            ax1.legend()
            ax1.set_title("Accuracy Graph")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Accuracy")
            ax1.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots(figsize=(7, 4))
            ax2.plot(hist['loss'],     label='Train Loss',      color='#4A90D9', linewidth=2)
            ax2.plot(hist['val_loss'], label='Validation Loss', color='#E74C3C', linewidth=2)
            ax2.legend()
            ax2.set_title("Loss Graph")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Loss")
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig2)
        except:
            st.info("ℹ️ Training graphs available after training. Save history.pkl from your training script.")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("Alzheimer prediction using MRI + clinical data | OASIS Dataset")