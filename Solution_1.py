import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed
tf.random.set_seed(42)

train_dir = r"C:\Users\user\Downloads\DeepLearningProject\X-Ray-Classifier\xrays\train"
test_dir = r"C:\Users\user\Downloads\DeepLearningProject\X-Ray-Classifier\xrays\test"

# 1. Data Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    shear_range=0.1,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=0.20 
)

test_datagen = ImageDataGenerator(rescale=1./255)

# 2. Data Loading
train_data = train_datagen.flow_from_directory(
    directory=train_dir,
    batch_size=32,
    target_size=(224, 224),
    class_mode="binary",
    subset="training",
    seed=42
)

valid_data = train_datagen.flow_from_directory(
    directory=train_dir,
    batch_size=32,
    target_size=(224, 224),
    class_mode="binary",
    subset="validation",
    seed=42
)

test_data = test_datagen.flow_from_directory(
    directory=test_dir,
    batch_size=32,
    target_size=(224, 224),
    class_mode="binary",
    shuffle=False
)

# 3. Functional API Architecture (The Fix)
base_model = EfficientNetB0(include_top=False, input_shape=(224, 224, 3), weights="imagenet")
base_model.trainable = False 

inputs = Input(shape=(224, 224, 3))
# CRITICAL: training=False prevents Batch Normalization from destroying pre-trained weights
x = base_model(inputs, training=False) 
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation="sigmoid")(x)

custom_model = Model(inputs, outputs)

# 4. Phase 1: Train Top Layers
custom_model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"]
)

early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

print("\n--- Starting Phase 1: Feature Extractor Training ---")
history_1 = custom_model.fit(train_data, validation_data=valid_data, epochs=15, callbacks=[early_stopping])

# 5. Phase 2: Fine-Tuning
print("\n--- Starting Phase 2: Fine-Tuning ---")
base_model.trainable = True

custom_model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    metrics=["accuracy"]
)

history_fine = custom_model.fit(train_data, validation_data=valid_data, epochs=10, callbacks=[early_stopping])

custom_model.save("xray_efficientnet_refined.keras")

# 6. Evaluation
test_loss, test_accuracy = custom_model.evaluate(test_data)
print(f"\nFinal Unbiased Test Loss: {test_loss:.4f}")
print(f"Final Unbiased Test Accuracy: {test_accuracy:.4f}")