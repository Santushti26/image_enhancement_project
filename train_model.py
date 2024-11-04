import os
import random
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.image as img

# Reading data function
def read_data(input_path, input_files, gt_path, gt_files, batch_size, size_input, num_channel, batch_no=None):
    data = []
    labels = []
    for i in range(batch_size):
        if batch_no is not None:
            r_idx = batch_no * batch_size + i
        else:
            r_idx = random.randint(0, len(input_files) - 1)

        image_path = os.path.join(input_path, input_files[r_idx])
        print(f"Reading file: {image_path}")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File not found: {image_path}")

        image = img.imread(image_path)
        if np.max(image) > 1:
            image = image / 255.0

        image = cv2.resize(image, (size_input, size_input))
        data.append(image)

        gt_image_path = os.path.join(gt_path, gt_files[r_idx])
        print(f"Reading ground truth file: {gt_image_path}")

        if not os.path.exists(gt_image_path):
            raise FileNotFoundError(f"File not found: {gt_image_path}")

        gt_image = img.imread(gt_image_path)
        if np.max(gt_image) > 1:
            gt_image = gt_image / 255.0

        gt_image = cv2.resize(gt_image, (size_input, size_input))
        labels.append(gt_image)

    return np.array(data), np.array(labels)


# Build the model using Keras API
def build_model(input_shape):
    inputs = tf.keras.layers.Input(shape=input_shape)
    conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    conv3 = tf.keras.layers.Conv2D(3, (3, 3), activation=None, padding='same')(conv2)
    return tf.keras.Model(inputs=inputs, outputs=conv3)


# Loss and metrics definitions
def compute_loss(true_labels, predicted):
    return tf.reduce_mean(tf.square(true_labels - predicted))  # MSE loss

def compute_ssim(true_labels, predicted):
    return tf.image.ssim(true_labels, predicted, max_val=1.0)

def compute_mse(true_labels, predicted):
    return tf.reduce_mean(tf.square(true_labels - predicted))


# Define some training parameters
input_path = "C:\\Users\\santu\\Desktop\\Codes\\Train\\train\\foggy"
gt_path = "C:\\Users\\santu\\Desktop\\Codes\\Train\\train\\clean"
validation_path = "C:\\Users\\santu\\Desktop\\Codes\\Train\\train\\val"
input_files = os.listdir(input_path)
gt_files = os.listdir(gt_path)
validation_files = os.listdir(validation_path)
batch_size = 25
patch_size = 64
num_channels = 3
learning_rate = 0.001
iterations = 50000
save_model_path = "./models"
model_name = "model_ckpt"

# Build the model
model = build_model((patch_size, patch_size, num_channels))

# Compile the model with optimizer, loss, and metrics
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Train and validate in a loop
for j in range(iterations):
    # Read training data
    train_data, train_label = read_data(input_path, input_files, gt_path, gt_files, batch_size, patch_size, num_channels)

    with tf.GradientTape() as tape:
        # Forward pass
        predictions = model(train_data, training=True)
        
        # Compute the loss
        loss = compute_loss(train_label, predictions)
        ssim = compute_ssim(train_label, predictions)
        mse = compute_mse(train_label, predictions)

    # Backward pass and optimization
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Print training progress
    print(f"Iteration {j+1}/{iterations}, Loss: {loss.numpy()}, SSIM: {ssim.numpy()}, MSE: {mse.numpy()}")

    # Validation every 100 iterations
    if (j + 1) % 100 == 0:
        print('Validation computation...')
        Validation_Loss = 0
        Validation_ssim = 0
        Validation_mse = 0
        validation_batches = int(len(validation_files) / batch_size)
        for batch in range(validation_batches):
            validation_data, validation_label = read_data(validation_path, validation_files, gt_path, gt_files, batch_size, patch_size, num_channels, batch)
            val_predictions = model(validation_data, training=False)

            val_loss = compute_loss(validation_label, val_predictions)
            val_ssim = compute_ssim(validation_label, val_predictions)
            val_mse = compute_mse(validation_label, val_predictions)

            Validation_Loss += val_loss / validation_batches
            Validation_ssim += val_ssim / validation_batches
            Validation_mse += val_mse / validation_batches

        print(f"Validation Loss: {Validation_Loss.numpy()}, SSIM: {Validation_ssim.numpy()}, MSE: {Validation_mse.numpy()}")

        # Save model checkpoint every 100 iterations
        model.save(f"{save_model_path}/{model_name}_{j+1}.h5")
