#!/usr/bin/env python3
import os
import datetime

import tensorflow as tf
import tensorflow.keras as keras
import Mnasnet

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255.0, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)

num_gpus = 2
batch_size = 100 * num_gpus
dataset_root = os.path.expanduser("~/datasets/imagenet")
train_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_root, "train"),
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode="categorical")
val_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_root, "val"),
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode="categorical")

# Load model
model = Mnasnet.MnasNet(input_shape=(224, 224, 3))
multigpu_model = keras.utils.multi_gpu_model(model, gpus=2)

multigpu_model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
multigpu_model.summary()

filepath = "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath,
    monitor="val_accuracy",
    verbose=1,
    save_best_only=True,
    mode="max")

print("Training started at", datetime.datetime.utcnow().isoformat())

# Train it
num_epochs = 50
multigpu_model.fit_generator(
    train_generator,
    steps_per_epoch=200000 // batch_size,
    epochs=num_epochs,
    validation_data=val_generator,
    validation_steps=8000 // batch_size,
    max_queue_size=batch_size * 3 // 2,  # 1.5x the batch size
    workers=8,
    callbacks=[checkpoint_callback])
# Evaluate it
# loss, acc = model.evaluate(x_test, y_test)
model.save_weights("final_model.hd5")
# print("Accuracy of: " + str(acc * 100.) + "%")
