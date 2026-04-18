import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D,
    Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0


def build_multimodal_model(input_shape_mri, input_shape_clinical, num_classes):

    # =================================
    # MRI INPUT
    # =================================
    mri_input = Input(shape=input_shape_mri, name="mri_input")

    # Convert grayscale → 3 channel
    x = Concatenate()([mri_input, mri_input, mri_input])

    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=x
    )

    # Freeze most layers (important for medical datasets)
    for layer in base_model.layers[-60:]:
       

       layer.trainable = True
    # Fine-tune last layers
    for layer in base_model.layers[-60:]:
        layer.trainable = True

    # MRI Feature Extraction
    x = GlobalAveragePooling2D()(base_model.output)
    x = GaussianNoise(0.05)(x)
    x = BatchNormalization()(x)

    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(128, activation="relu")(x)
    mri_features = Dropout(0.4)(x)

    # =================================
    # CLINICAL BRANCH
    # =================================
    clinical_input = Input(shape=(input_shape_clinical,), name="clinical_input")

    y = Dense(128, activation="relu")(clinical_input)
    y = BatchNormalization()(y)
    y = Dropout(0.4)(y)

    y = Dense(64, activation="relu")(y)
    y = BatchNormalization()(y)
    clinical_features = Dropout(0.3)(y)

    # =================================
    # MULTIMODAL FUSION
    # =================================
    combined = Concatenate()([mri_features, clinical_features])

    z = Dense(256, activation='relu')(combined)
    z = BatchNormalization()(z)
    z = Dropout(0.5)(z)

    z = Dense(128, activation='relu')(z)
    z = Dropout(0.4)(z)

    output = Dense(num_classes, activation='softmax')(z)


    # CREATE MODEL
    model = Model(inputs=[mri_input, clinical_input], outputs=output)

# COMPILE MODEL
    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

    return model


