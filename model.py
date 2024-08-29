import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Activation, Input, BatchNormalization, Add, Concatenate, UpSampling2D, Multiply
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Function to apply a sharpening filter
def sharpen_image(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

# Prepare the dataset
def load_images_from_folder(folder, target_size=(256, 256)):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, target_size)  # Resize all images to the same size
            images.append(img)
    return np.array(images)

# Load your denoised dataset
input_folder = r'C:\Users\logam\OneDrive\Documents\isro\image'
output_folder = r'C:\Users\logam\OneDrive\Documents\isro\final'
os.makedirs(output_folder, exist_ok=True)

images = load_images_from_folder(input_folder)
images = images.astype('float32') / 255.0  # Normalize images

# Define the Retinex-inspired enhancement model
def build_retinex_enhancement_model(input_shape):
    input_img = Input(shape=input_shape)

    # First convolutional block
    x1 = Conv2D(64, (3, 3), padding='same')(input_img)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    # Second convolutional block
    x2 = Conv2D(64, (3, 3), padding='same')(x1)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)

    # Adding more convolutional layers for better feature extraction
    x3 = Conv2D(64, (3, 3), padding='same')(x2)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)

    # Attention mechanism to focus on important features
    x4 = Conv2D(64, (3, 3), padding='same')(x3)
    x4 = BatchNormalization()(x4)
    x4 = Activation('sigmoid')(x4)
    attention = Multiply()([x3, x4])

    # Residual connection for better gradient flow
    x5 = Add()([x2, attention])

    # Output layer
    output_img = Conv2D(3, (3, 3), padding='same')(x5)
    output_img = Activation('sigmoid')(output_img)

    return Model(inputs=input_img, outputs=output_img)

# Build the model
input_shape = images[0].shape
model = build_retinex_enhancement_model(input_shape)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Setup checkpoint to save the best model
checkpoint = ModelCheckpoint('retinex_enhanced_model.h5', monitor='loss', save_best_only=True, mode='min', verbose=1)

# Train the model
model.fit(images, images, epochs=100, batch_size=8, callbacks=[checkpoint])

# Load the best model
model.load_weights('retinex_enhanced_model.h5')

# Enhance images using the model and additional sharpening
for i, img in enumerate(images):
    enhanced_img = model.predict(np.expand_dims(img, axis=0))[0]
    enhanced_img = sharpen_image(enhanced_img * 255.0).astype(np.uint8)
    cv2.imwrite(os.path.join(output_folder, f"enhanced_{i}.png"), enhanced_img)

print(f"Enhanced images saved to {output_folder}")