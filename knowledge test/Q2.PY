import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Initialize the ImageDataGenerator with augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=40,          # Randomly rotate images by up to 40 degrees
    width_shift_range=0.2,      # Randomly shift images horizontally by up to 20% of the width
    height_shift_range=0.2,     # Randomly shift images vertically by up to 20% of the height
    shear_range=0.2,            # Apply random shearing transformations
    zoom_range=0.2,             # Randomly zoom in or out on images
    horizontal_flip=True,       # Randomly flip images horizontally
    fill_mode='nearest'         # Fill in missing pixels after transformation
)

# Load an example image
from tensorflow.keras.preprocessing import image
import numpy as np

img = image.load_img('cat.jpg')  # Replace with your image path
img_array = image.img_to_array(img)         # Convert image to array
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Generate augmented images
augmented_images = datagen.flow(img_array, batch_size=1)

# Plot original and augmented images
fig, ax = plt.subplots(1, 4, figsize=(15, 5))
for i, batch in enumerate(augmented_images):
    if i >= 3:
        break
    ax[i].imshow(np.squeeze(batch[0]).astype('uint8'))
    ax[i].axis('off')
plt.show()
