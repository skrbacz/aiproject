import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# configuration and setup
base_dir = 'D:/projects/aiproject/aiproject/processed_images'

# model hyperparameters
img_size = (150, 150)
batch_size = 32
epochs = 19

# check if the base directory exists
if not os.path.isdir(os.path.join(base_dir, 'train')):
    print(f"ERROR: '{base_dir}' doesn't contain 'train' and 'test' folders")

# data loading and preprocessing
print("Loading data...")

train_data = tf.keras.utils.image_dataset_from_directory(
    directory=os.path.join(base_dir, 'train'),
    labels='inferred',
    label_mode='binary',
    image_size=img_size,
    interpolation='nearest',
    batch_size=batch_size,
    shuffle=True
)

test_data = tf.keras.utils.image_dataset_from_directory(
    directory=os.path.join(base_dir, 'test'),
    labels='inferred',
    label_mode='binary',
    image_size=img_size,
    interpolation='nearest',
    batch_size=batch_size,
    shuffle=False
)

# normalize the pixel values
def process(image, label):
    # center crop to remove sclera
    crop_size = 112 
    offset = (img_size[0] - crop_size) // 2 
    
    image = tf.image.crop_to_bounding_box(
        image, 
        offset,          # offset_height
        offset,          # offset_width
        crop_size,       # target_height
        crop_size        # target_width
    )
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_data = train_data.map(process)
test_data = test_data.map(process)

print(f"Found {len(train_data) * batch_size} training examples")
print(f"Found {len(test_data) * batch_size} test examples")


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

input_shape = img_size + (3,)
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
    train_data,
    epochs=epochs,
    validation_data=test_data,
    callbacks=[early_stopping] 
)

# evaluation
print("\nevaluating model performance on test set...")

loss, accuracy = model.evaluate(test_data)

print(f"test loss: {loss:.4f}")
print(f"test accuracy: {accuracy*100:.2f}%")

# save model
model_path = 'cataract_classifier.h5'
model.save(model_path)
print(f"\nmodel saved to {model_path}")
