import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers.legacy import Adam  # ✅ Use legacy Adam in TF 2.19
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import numpy as np
import os
import json
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


# ✅ Enable GPU Memory Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# ✅ Dataset Paths
DATASET_30S = "data/fingerprint_30s/"
DATASET_FULL = "data/fingerprints/"  # ✅ Full song dataset path
MERGED_DATASET = "filtered_fingerprint_merged/"  # ✅ Combined dataset

# ✅ Select Genres
available_genres = [d for d in os.listdir(DATASET_30S) if os.path.isdir(os.path.join(DATASET_30S, d))]
selected_genres = input("Enter genres to train on (comma-separated): ").strip().split(",")
selected_genres = [g.strip() for g in selected_genres if g.strip() in available_genres]

if not selected_genres:
    print("⚠️ No valid genres selected. Training on all available genres.")
    selected_genres = available_genres

# ✅ Merge 30s and Full Song Data
def merge_datasets():
    if os.path.exists(MERGED_DATASET):
        shutil.rmtree(MERGED_DATASET)
    os.makedirs(MERGED_DATASET, exist_ok=True)
    
    for genre in selected_genres:
        os.makedirs(os.path.join(MERGED_DATASET, genre), exist_ok=True)
        
        # ✅ Copy 30s clips
        if os.path.exists(os.path.join(DATASET_30S, genre)):
            shutil.copytree(os.path.join(DATASET_30S, genre), os.path.join(MERGED_DATASET, genre), dirs_exist_ok=True)
        
        # ✅ Copy Full Songs
        if os.path.exists(os.path.join(DATASET_FULL, genre)):
            shutil.copytree(os.path.join(DATASET_FULL, genre), os.path.join(MERGED_DATASET, genre), dirs_exist_ok=True)

merge_datasets()

# ✅ Data Augmentation
img_size = 128
batch_size = 64

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2, 
    shear_range=0.2, 
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode="nearest",
    validation_split=0.2
)

# ✅ Load Data
train_data = datagen.flow_from_directory(MERGED_DATASET, target_size=(img_size, img_size), 
                                        batch_size=batch_size, class_mode="categorical", subset="training")
val_data = datagen.flow_from_directory(MERGED_DATASET, target_size=(img_size, img_size), 
                                       batch_size=batch_size, class_mode="categorical", subset="validation")

# ✅ Save Labels
LABELS = list(train_data.class_indices.keys())
with open("model/labels.json", "w") as f:
    json.dump(LABELS, f)

# ✅ Load Model
base_model = tf.keras.applications.MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights="imagenet")
for layer in base_model.layers[:100]:
    layer.trainable = False

x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.02))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(len(LABELS), activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=x)
model.compile(optimizer=Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=['accuracy'])

# ✅ Callbacks
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
]

# ✅ Train Model
history = model.fit(train_data, validation_data=val_data, epochs=80, callbacks=callbacks)

# ✅ Save Model
model.save("model/merged_audio_model.keras")
print("✅ Merged Model (30s + Full Songs) saved successfully!")

# ✅ Plot Training Curves
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy', color='green')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Curve')
plt.legend()
plt.savefig("accuracy_curve_merged.png")
plt.show()

# ✅ Confusion Matrix & Classification Report
def evaluate_model():
    y_true, y_pred = [], []
    for i in range(len(val_data)):
        X_batch, y_batch = val_data[i]
        y_batch_pred = model.predict(X_batch)
        y_true.extend(np.argmax(y_batch, axis=1))
        y_pred.extend(np.argmax(y_batch_pred, axis=1))
    
    report = classification_report(y_true, y_pred, target_names=LABELS, digits=4)
    print(report)
    with open("classification_report_merged.txt", "w") as f:
        f.write(report)
    
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix_merged.png")
    plt.show()

evaluate_model()

