import tensorflow as tf
from model import create_model
from data_preprocessing import get_data_generators

# Load data
train_dir = 'data/train'
val_dir = 'data/val'
train_generator, val_generator = get_data_generators(train_dir, val_dir)

# Define and compile model
model = create_model()
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size
)

# Save the trained model
model.save('saved_models/model.h5')
