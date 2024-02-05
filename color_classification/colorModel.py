import numpy as np
import tensorflow as tf
import pandas as pd
from PIL import Image
from keras import layers, models
from keras.utils import to_categorical

train_directory = './train/dataset'
# change file name here
train_df = pd.read_csv(train_directory + '/labels.txt')

train_file_paths = train_df['file'].values
train_letter_colors = train_df[' letter_color'].values
train_shape_colors = train_df[' shape_color'].values

image_list = np.array([np.array(Image.open(train_directory + file_path))/255.0 for file_path in train_file_paths])

num_classes = 8
train_shape_colors_onehot = to_categorical(train_shape_colors, num_classes=num_classes)
train_letter_colors_onehot = to_categorical(train_letter_colors, num_classes=num_classes)

input_layer = layers.Input(shape=(128, 128, 3))

# Shared convolutional layers
conv1 = layers.Conv2D(64, (3, 3), activation='relu')(input_layer)
conv2 = layers.Conv2D(128, (3, 3), activation='relu')(conv1)

# Shape prediction branch
shape_flatten = layers.Flatten()(conv2)
shape_dense = layers.Dense(64, activation='relu')(shape_flatten)
shape_output = layers.Dense(num_classes, activation='softmax', name='shape_output')(shape_dense)

# Letter prediction branch
letter_flatten = layers.Flatten()(conv2)
letter_dense = layers.Dense(64, activation='relu')(letter_flatten)
letter_output = layers.Dense(num_classes, activation='softmax', name='letter_output')(letter_dense)

# Create the model
model = models.Model(inputs=input_layer, outputs=[shape_output, letter_output])

model.compile(optimizer='adam',
              loss={'shape_output': 'categorical_crossentropy', 'letter_output': 'categorical_crossentropy'},
              metrics={'shape_output': 'accuracy', 'letter_output': 'accuracy'})

# Train the model
model.fit(x=image_list, y={'shape_output': train_shape_colors_onehot, 'letter_output': train_letter_colors_onehot},
          epochs=1)

model.save("./trained_model")