import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

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



# 3. Corrected CNN Architecture

model = tf.keras.models.Sequential([

    # Block 1

    tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(224, 224, 3)),

    tf.keras.layers.MaxPool2D(2),

   

    # Block 2

    tf.keras.layers.Conv2D(64, 3, activation="relu"),

    tf.keras.layers.MaxPool2D(2),

   

    # Block 3

    tf.keras.layers.Conv2D(128, 3, activation="relu"),

    tf.keras.layers.MaxPool2D(2),

   

    # Block 4

    tf.keras.layers.Conv2D(128, 3, activation="relu"),

    tf.keras.layers.MaxPool2D(2),

   

    # Classification Head

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dropout(0.5), # Crucial for training from scratch

    tf.keras.layers.Dense(256, activation="relu"), # Replaced sigmoid with relu

    tf.keras.layers.Dense(1, activation="sigmoid")

])



model.compile(

    loss="binary_crossentropy",

    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),

    metrics=["accuracy"]

)



early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True

)



# 4. Training

history_2 = model.fit(

    train_data,

    validation_data=valid_data,

    epochs=20, # Increased epochs since early stopping will catch it

    callbacks=[early_stopping]

)



# 5. Visualization & Evaluation

history_df = pd.DataFrame(history_2.history)

history_df.loc[:, ['loss', 'val_loss', 'accuracy', 'val_accuracy']].plot(title="Custom CNN Training Curves")

plt.show()



test_loss, test_accuracy = model.evaluate(test_data)

print(f"Valid Test Loss: {test_loss:.4f}")

print(f"Valid Test Accuracy: {test_accuracy:.4f}")