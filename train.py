import pandas as pd
import os
import numpy as np
import cv2
import nibabel as nib
import tensorflow as tf
import pickle
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from alzheimers_multimodal_model import build_multimodal_model

IMG_SIZE = 160
EPOCHS = 120
BATCH_SIZE = 16
NUM_CLASSES = 3

# =========================================
# Paths
# =========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
mri_path = os.path.join(BASE_DIR, "data", "mri")

# =========================================
# Load Clinical Data
# =========================================
df = pd.read_excel("oasis_longitudinal_demographics.xlsx")

df = df[['Subject ID','Age','MMSE','eTIV','nWBV','ASF','SES','M/F','CDR']]
df = df.dropna()

df['Gender'] = df['M/F'].map({'M':1,'F':0})

# ✅ FIX — safe map_label that handles any CDR value
def map_label(cdr):
    try:
        cdr = float(cdr)
        if cdr == 0:
            return 0
        elif cdr == 0.5:
            return 1
        else:
            return 2
    except:
        return None   # bad row → will be dropped

df['label'] = df['CDR'].apply(map_label)
df = df.dropna(subset=['label'])           # ✅ drops bad rows
df['label'] = df['label'].astype(int)      # ✅ ensures integer labels

# =========================================
# Collect Subjects
# =========================================
subjects = os.listdir(mri_path)

valid_subjects = []
for subject in subjects:
    subject_id = subject.split('_MR')[0].strip()
    if subject_id in df['Subject ID'].values:
        valid_subjects.append(subject)

print("Total valid subjects:", len(valid_subjects))


# =========================================
# Train Test Split
# =========================================
train_subjects, test_subjects = train_test_split(
    valid_subjects,
    test_size=0.2,
    random_state=42
)

print("Train subjects:", len(train_subjects))
print("Test subjects:",  len(test_subjects))

# =========================================
# Load MRI + Clinical
# =========================================
def load_data(subject_list):
    mri_data = []
    clinical_data = []
    labels = []

    for subject in subject_list:

        subject_path = os.path.join(mri_path, subject, "RAW")

        if not os.path.exists(subject_path):
            continue

        subject_id = subject.split('_MR')[0].strip()
        row = df[df['Subject ID'] == subject_id]

        if len(row) == 0:
            continue

        label = row['label'].values[0]

        age    = row['Age'].values[0]
        mmse   = row['MMSE'].values[0]
        etiv   = row['eTIV'].values[0]
        nwbv   = row['nWBV'].values[0]
        asf    = row['ASF'].values[0]
        ses    = row['SES'].values[0]
        gender = row['Gender'].values[0]

        clinical_features = [age, mmse, etiv, nwbv, asf, ses, gender]

        for file in os.listdir(subject_path):
            if file.endswith(".img"):

                img_path = os.path.join(subject_path, file)

                img = nib.load(img_path).get_fdata()

                depth  = img.shape[2]
                center = depth // 2

                # Use center slices (best for brain MRI)
                for i in range(max(0, center-12), min(depth, center+13)):
                    slice_img = img[:,:,i]

                    slice_img = cv2.resize(slice_img,(IMG_SIZE,IMG_SIZE)).astype(np.float32)

                    slice_img = (slice_img - np.min(slice_img)) / (
                        np.max(slice_img) - np.min(slice_img) + 1e-8
                    )

                    mri_data.append(slice_img)
                    clinical_data.append(clinical_features)
                    labels.append(label)

    mri_data      = np.array(mri_data, dtype=np.float32).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    clinical_data = np.array(clinical_data)
    labels        = np.array(labels)

    return mri_data, clinical_data, labels

# =========================================
# Load Train Test Data
# =========================================
X_train, clinical_train, y_train = load_data(train_subjects)
X_test,  clinical_test,  y_test  = load_data(test_subjects)

print("MRI train shape:",      X_train.shape)
print("Clinical train shape:", clinical_train.shape)
print("Labels shape:",         y_train.shape)

# =========================================
# Normalize Clinical Features
# =========================================
scaler = StandardScaler()

clinical_train = scaler.fit_transform(clinical_train)
clinical_test  = scaler.transform(clinical_test)

joblib.dump(scaler, "clinical_scaler.pkl")

# =========================================
# Compute Class Weights
# =========================================
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weight_dict = dict(enumerate(class_weights))

# =========================================
# Shuffle Training Data
# =========================================
idx = np.random.permutation(len(X_train))

X_train        = X_train[idx]
clinical_train = clinical_train[idx]
y_train        = y_train[idx]

# =========================================
# Build Model
# =========================================
model = build_multimodal_model(
    X_train.shape[1:],
    clinical_train.shape[1],
    NUM_CLASSES
)

# =========================================
# Callbacks
# =========================================
early_stop = EarlyStopping(
    monitor='val_accuracy',   # ← monitor accuracy not loss
    patience=15,              # ← give more time
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=6,               # ← reduce LR less aggressively
    min_lr=1e-6,
    verbose=1
)
# =========================================
# Train
# =========================================
history = model.fit(
    [X_train, clinical_train],
    y_train,
    validation_data=([X_test, clinical_test], y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weight_dict,
    shuffle=True,
    verbose=1
)

# =========================================
# Evaluate
# =========================================
print("\nEvaluating model with subject voting...")

pred         = model.predict([X_test, clinical_test])
pred_classes = np.argmax(pred, axis=1)

correct = 0
total   = 0

slices_per_subject = 25

for i in range(0, len(pred_classes), slices_per_subject):
    subject_preds = pred_classes[i:i+slices_per_subject]

    # majority vote
    final_pred = np.bincount(subject_preds).argmax()
    true_label = y_test[i]

    if final_pred == true_label:
        correct += 1

    total += 1

subject_accuracy = correct / total
print(f"\nFINAL SUBJECT ACCURACY: {subject_accuracy*100:.2f}%")

# =========================================
# Save Model
# =========================================
model.save("alzheimers_model.keras")

history_df = pd.DataFrame(history.history)
history_df.to_csv("training_history.csv", index=False)

print("Model and training history saved.")

# =========================================
# Save files for Streamlit app
# =========================================
np.save("X_test_mri.npy",      X_test)
np.save("X_test_clinical.npy", clinical_test)
np.save("y_test.npy",          y_test)

with open("history.pkl", "wb") as f:
    pickle.dump(history.history, f)

print("✅ Saved: X_test_mri.npy")
print("✅ Saved: X_test_clinical.npy")
print("✅ Saved: y_test.npy")
print("✅ Saved: history.pkl")
print("✅ Run: streamlit run alzheimer_app.py")