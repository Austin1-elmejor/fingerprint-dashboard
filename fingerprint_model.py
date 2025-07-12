import tensorflow as tf
from tensorflow.keras import layers, models

# === Dataset Setup ===
dataset_path = "Scanned"  # should contain subfolders like A+/, B+/, O-/, etc.
img_size = (128, 128)
batch_size = 32
seed = 123

# Load the dataset first
train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=seed,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="categorical"
)

val_ds_raw = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=seed,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="categorical"
)

# Extract class names before transformation
num_classes = len(train_ds_raw.class_names)

# Normalize and prepare datasets
AUTOTUNE = tf.data.AUTOTUNE
normalization_layer = layers.Rescaling(1. / 255)
train_ds = train_ds_raw.map(lambda x, y: (normalization_layer(x), y)).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds_raw.map(lambda x, y: (normalization_layer(x), y)).prefetch(buffer_size= AUTOTUNE)


# Improve performance with prefetch
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# === Define CNN Model ===
num_classes = len(train_ds_raw.class_names)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# === Compile & Train ===
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds, validation_data=val_ds, epochs=10)

# === Save Model ===
model.save("fingerprint_model.h5")
print("✅ Model saved as fingerprint_model.h5")

