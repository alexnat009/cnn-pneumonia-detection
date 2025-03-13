import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_dims = 128
batch_size = 64
input_path = "C:/Users/aleksandre/PycharmProjects/data-flair-projects/DeepLearningPneumoniaDetection/dataset/chest_xray"

# creating model
model = Sequential()
model.add(layer=Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(image_dims, image_dims, 3)))
model.add(layer=MaxPool2D(pool_size=(2, 2)))
model.add(layer=Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(layer=MaxPool2D(pool_size=(2, 2)))
model.add(layer=Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(layer=MaxPool2D(pool_size=(2, 2)))
model.add(layer=Flatten())
model.add(layer=Dense(units=128, activation='relu'))
model.add(layer=Dense(units=1, activation='sigmoid'))
# model.summary()

# compiling
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

training_data_generator = ImageDataGenerator(rescale=1 / 255., shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

training_gen = training_data_generator.flow_from_directory(directory=f'{input_path}/train',
                                                           target_size=(image_dims, image_dims),
                                                           batch_size=batch_size, class_mode='binary')
validation_data_generator = ImageDataGenerator(rescale=1 / 255.)
validation_gen = validation_data_generator.flow_from_directory(directory=f'{input_path}/val',
                                                               target_size=(image_dims, image_dims),
                                                               batch_size=batch_size,
                                                               class_mode='binary')
epochs = 10
history = model.fit_generator(generator=training_gen,
                              steps_per_epoch=10, epochs=epochs + 2,
                              validation_data=validation_gen,
                              validation_steps=validation_gen.samples)

# visualising model accuracy and validation accuracy
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 6))
ax[0].set_title('Accuracy scores')
ax[0].plot(history.history['accuracy'])
ax[0].plot(history.history['val_accuracy'])
ax[0].legend(['accuracy', 'val_accuracy'])
ax[1].set_title('Loss value')
ax[1].plot(history.history['loss'])
ax[1].plot(history.history['val_loss'])
ax[1].legend(['loss', 'val_loss'])
plt.show()

test_data_generator = ImageDataGenerator(rescale=1 / 255.)

test_gen = test_data_generator.flow_from_directory(directory=f'{input_path}/test',
                                                   target_size=(image_dims, image_dims),
                                                   batch_size=128,
                                                   class_mode='binary')

eval_result = model.evaluate(generator=test_gen, steps=100)
print(f'loss rate at evaluation data : {eval_result[0]}')
print(f'accuracy rate at evaluation data : {eval_result[1]}')
