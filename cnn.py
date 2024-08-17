import cv2
import numpy as np
import matplotlib.pyplot as plt

#hazy image 
# Load a sample image
image = cv2.imread('sample_image.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize image for consistency
image = cv2.resize(image, (256, 256))

# Display the original image
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')
plt.show()



from skimage import filters

# Gaussian Blurring function
def gaussian_blur(img, sigma=1):
    img_blur = filters.gaussian(img, sigma=sigma, multichannel=True)
    return img_blur
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure, filters

# Histogram Equalization function
def histogram_equalization(img):
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    r_eq = exposure.equalize_hist(r)
    g_eq = exposure.equalize_hist(g)
    b_eq = exposure.equalize_hist(b)
    img_eq = np.stack((r_eq, g_eq, b_eq), axis=-1)
    return img_eq

if image is None:
    print(f"Error: Unable to load image at {image_path}")
else:
    # Convert the image to RGB format (OpenCV loads images in BGR by default)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize image for consistency
    image = cv2.resize(image, (256, 256))

    # Proceed with enhancement
    img_hist_eq = histogram_equalization(image)
    img_gaussian_blur = gaussian_blur(image, sigma=2)



if image is None:
    print(f"Error: Unable to load image at {image_path}")
else:
    # Convert the image to RGB format (OpenCV loads images in BGR by default)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize image for consistency
    image = cv2.resize(image, (256, 256))

    # Proceed with enhancement
    img_hist_eq = histogram_equalization(image)
    img_gaussian_blur = gaussian_blur(image, sigma=2)



from skimage import filters

# Revised Gaussian Blurring function
def gaussian_blur(img, sigma=1):
    # Split the image into its Red, Green, and Blue channels
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    
    # Apply Gaussian blur to each channel
    r_blur = filters.gaussian(r, sigma=sigma)
    g_blur = filters.gaussian(g, sigma=sigma)
    b_blur = filters.gaussian(b, sigma=sigma)
    
    # Merge the blurred channels back together
    img_blur = np.stack((r_blur, g_blur, b_blur), axis=-1)
    
    return img_blur


# Example of using the functions
image = cv2.imread('sample_image.jpg')  # Replace with your image file path
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (256, 256))

img_hist_eq = histogram_equalization(image)
img_gaussian_blur = gaussian_blur(image, sigma=2)

# Display the results
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(img_hist_eq)
ax[0].set_title("Histogram Equalization")
ax[0].axis('off')
ax[1].imshow(img_gaussian_blur)
ax[1].set_title("Gaussian Blur")
ax[1].axis('off')
plt.show()

import tensorflow as tf
from tensorflow.keras import layers, models

# Define a simple model
def create_simple_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Create the model
model = create_simple_model()
model.summary()


# Hybrid Enhancement Function
def hybrid_enhancement(image, model):
    #Apply classical methods
    image_classical = histogram_equalization(image)
    image_classical = gaussian_blur(image_classical, sigma=2)
    
    #  Pass the image through the deep learning model
    image_dl_input = np.expand_dims(image_classical, axis=0)
    image_dl_output = model.predict(image_dl_input)[0]
    
    return image_dl_output

# Assuming you have trained the model (for demo purposes we skip training)
# Enhanced Image using the Hybrid Method
enhanced_image = hybrid_enhancement(image, model)

# Display the enhanced image
plt.imshow(enhanced_image)
plt.title("Hybrid Enhanced Image")
plt.axis('off')
plt.show()



