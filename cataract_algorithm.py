import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# configuration and setup

BASE_DIR = 'D:/projects/aiproject/aiproject/processed_images'

# model hyperparameters
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 19

# check if the base directory exists
if not os.path.isdir(os.path.join(BASE_DIR, 'train')):
    print(f"ERROR: The directory '{BASE_DIR}' does not seem to contain 'train' and 'test' folders.")
    print("Please update the 'BASE_DIR' variable to the correct path.")

# data loading and preprocessing

print("Loading data...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    directory=os.path.join(BASE_DIR, 'train'),
    labels='inferred',
    label_mode='binary',
    image_size=IMG_SIZE,
    interpolation='nearest',
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    directory=os.path.join(BASE_DIR, 'test'),
    labels='inferred',
    label_mode='binary',
    image_size=IMG_SIZE,
    interpolation='nearest',
    batch_size=BATCH_SIZE,
    shuffle=False
)

# normalize the pixel values
def process(image, label):
    # center crop to remove sclera
    crop_size = 112 
    offset = (IMG_SIZE[0] - crop_size) // 2 
    
    image = tf.image.crop_to_bounding_box(
        image, 
        offset,          # offset_height
        offset,          # offset_width
        crop_size,       # target_height
        crop_size        # target_width
    )
    
    image = tf.image.resize(image, IMG_SIZE)
    
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_ds = train_ds.map(process)
test_ds = test_ds.map(process)

print(f"Found {len(train_ds) * BATCH_SIZE} training examples.")
print(f"Found {len(test_ds) * BATCH_SIZE} test examples.")


# model def

def build_cnn_model(input_shape):
    model = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomContrast(0.2),
        layers.RandomHue(0.1),
        
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),

        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
    
        layers.Flatten(),
        
        
        layers.Dropout(0.5),

        
        layers.Dense(128, activation='relu'),
        
        
        layers.Dense(1, activation='sigmoid')
    ])
    return model

input_shape = IMG_SIZE + (3,)
model = build_cnn_model(input_shape)

# compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()


# training

print("\nStarting model training...")

# early stopping to stop training if the validation loss doesnt improve
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=test_ds,
    callbacks=[early_stopping] 
)

# evaluation

print("\nEvaluating model performance on the test set...")

loss, accuracy = model.evaluate(test_ds)

print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")

# save model

MODEL_SAVE_PATH = 'cataract_classifier.h5'
model.save(MODEL_SAVE_PATH)
print(f"\nModel saved successfully to {MODEL_SAVE_PATH}")