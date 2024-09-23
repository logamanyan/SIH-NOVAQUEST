opencv-python

import os
import numpy as np
import cv2
from google.colab import files
from google.colab import drive
import zipfile
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, Add
from tensorflow.keras.optimizers import Adam
from tqdm.notebook import tqdm

# Mount Google Drive
drive.mount('/content/drive')

def sharpen_image(img):
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)

def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames

def build_sharpening_model(input_shape):
    input_img = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(input_img)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(3, (3, 3), padding='same')(x)
    output_img = Add()([input_img, x])
    return Model(inputs=input_img, outputs=output_img)

# Create folders in Google Drive
input_folder = '/content/sample_data/input'
output_folder = '/content/sample_data/output'
os.makedirs(input_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# Upload images
uploaded = files.upload()
for filename, content in uploaded.items():
    with open(os.path.join(input_folder, filename), 'wb') as f:
        f.write(content)

print(f"Uploaded images saved to {input_folder}")

# Load your dataset
images, filenames = load_images_from_folder(input_folder)

# Normalize images
normalized_images = [img.astype('float32') / 255.0 for img in images]

# Build and compile the model
input_shape = images[0].shape
model = build_sharpening_model(input_shape)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train the model
for _ in tqdm(range(50), desc="Training Progress"):  # Using tqdm for progress bar
    for img in normalized_images:
        model.fit(np.expand_dims(img, axis=0), np.expand_dims(img, axis=0), epochs=1, verbose=0)

# Enhance and save images
for img, filename in tqdm(zip(images, filenames), total=len(images), desc="Enhancing Images"):
    # Normalize the input image
    norm_img = img.astype('float32') / 255.0

    # Apply the model
    enhanced_img = model.predict(np.expand_dims(norm_img, axis=0))[0]

    # Denormalize
    enhanced_img = (enhanced_img * 255).astype(np.uint8)

    # Apply additional sharpening
    sharpened_img = sharpen_image(enhanced_img)

    # Ensure the output is in the correct range
    final_img = np.clip(sharpened_img, 0, 255).astype(np.uint8)

    # Save the enhanced image
    cv2.imwrite(os.path.join(output_folder, f"enhanced_{filename}"), final_img)

print(f"Enhanced images saved to {output_folder}")

# Create a zip file of the output folder
zip_path = '/content/sample_data/Untitled Folder/enhanced_imqges.zip'
with zipfile.ZipFile(zip_path, 'w') as zipf:
    for root, dirs, files in os.walk(output_folder):
        for file in files:
            zipf.write(os.path.join(root, file),
                       os.path.relpath(os.path.join(root, file),
                                       os.path.join(output_folder, '..')))

# Download the zip file
files.download(zip_path)
