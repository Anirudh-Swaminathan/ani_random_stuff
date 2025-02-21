#!/usr/bin/env python3
"""
Program to utilize multiple GPUs, if available with Tensorflow Distributed Training

This is single host, multiple GPU (so, 1 machine with 2 or more GPUs)
Each GPU will run a REPLICA of the model
This is Data Parallelism.
Each mini-batch is split into multiple local mini-batches equally across all GPUs.
Each GPU will run on its local mini-batch a forward pass, and a backward pass.
Weight updates from local gradients are efficiently merged across the 8 replicas.
Replicas always stay in sync, since every single individual weight variable is a
Mirrored Variable Object
"""
# native imports here
import os

# 3rd party imports here
import tensorflow as tf
import keras
import tensorflow_datasets as tfds


def normalize_img(image, label):
    """
    Normalizes images: `uint8` -> `float32`

    @param image tf.uint8 image to normalize
    @param label tf label

    @return tuple (tf.float32 normalized image, label)
    """
    img = tf.cast(image, tf.float32) / 255.0
    return tf.reshape(img, (784,)), label

def get_compiled_model():
    """
    Returns compiled model

    @return model
    """
    # Make a simple 2-layer densely-connected neural network.
    # define input layer
    inputs = keras.Input(shape=(784,), name='ani_input')

    # Fully connected Hidden layer 1 with 256 neurons,
    # each with ReLU activation connected to input layer
    x = keras.layers.Dense(256, activation="relu", name="ani_hidden_1")(inputs)

    # Fully connected Hidden layer 2 with 256 neurons,
    # each with ReLU activation connected to Hidden layer 1
    x = keras.layers.Dense(256, activation="relu", name="ani_hidden_2")(x)

    # Final output layer with 10 neurons for output layer fully connected to hidden layer 2
    # we output logits, not probabilities
    outputs = keras.layers.Dense(10, name="ani_output")(x)

    # wrap the series of connected layers inside keras.Model
    model = keras.Model(inputs=inputs, outputs=outputs, name="ani_model")

    # compile model with Adam optimizer,
    # SparseCategoricalCrossEntropy (So classification loss after implicit Softmax),
    # and metric as SparseCategoricalAccuracy()
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


def get_TFDS_mnist():
    """
    Returns MNIST dataset from TFDS

    @return (train, val, test) tuple of datasets
    """
    # define batch size
    batch_size = 500

    # define number of validation samples
    num_val_samples = 10000

    # Load MNIST dataset
    # shuffle_files=True ensures shuffling when training
    # as_supervised=True returns a tuple of (img, label),
    # rather than a dictionary of {'image': img, 'label': label}
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    # TFDS provide images of type tf.uint8, while the model expects tf.float32
    # Therefore, I need to normalize images using tf.data.Dataset.map
    ds_train = ds_train.map(
        normalize_img,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # cache dataset before shuffling for a better performance as I fit the dataset in memory
    ds_train = ds_train.cache()

    # For true randomness, set shuffle buffer to full dataset size
    # for large datasets that can't fit in memory,
    # can use buffer_size=1000 if my system allows it
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)


    # Batch elements of the dataset after shuffling to get unique batches at each epoch
    ds_train = ds_train.batch(batch_size)

    # Good practice to end the pipeline by prefetching for performance
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    # Build an evaluation pipeline
    # testing pipeline is similar to training pipeline, with small differences
    # First, normalize the test set, the same way as training set
    ds_test = ds_test.map(
        normalize_img,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # No need to call shuffle on test set
    # Batch before calling cache() since the batches can be same between epochs
    ds_test = ds_test.batch(batch_size)

    # Caching is done after batching because batches can be the same between epochs
    ds_test = ds_test.cache()

    # prefetch for improved performance
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    # do the train, val, test splitting now
    num_total_samples = ds_info.splits['train'].num_examples
    num_train_samples = num_total_samples - num_val_samples
    print(f"Number of training samples: {num_train_samples}")
    ds_tr = ds_train.take(num_train_samples)
    ds_val = ds_train.skip(num_train_samples).take(num_val_samples)
    print(f"Number of validation samples: {num_val_samples}")
    print(f"Number of test samples: {ds_info.splits['test'].num_examples}")

    return (
        ds_tr,
        ds_val,
        ds_test,
        batch_size
    )

def get_dataset():
    """
    Returns MNIST dataset

    @return (train, val, test) tuple of datasets
    """
    # define batch size
    batch_size = 50

    # define number of validation samples
    num_val_samples = 10000

    # Return the MNIST dataset in the form of a 'tf.data.Dataset' object.
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Preprocess the data (these are Numpy arrays)
    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 784).astype("float32") / 255.0
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    # Reserve the final num_val_samples samples for validation
    x_val = x_train[-num_val_samples:]
    y_val = y_train[-num_val_samples:]
    x_train = x_train[:-num_val_samples]
    y_train = y_train[:-num_val_samples]
    return (
        tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size),
        batch_size
    )

def make_or_restore_model(checkpoint_dir):
    """
    Makes or restores a model

    @param checkpoint_dir str the checkpoint directory to restore from

    @return latest model
    """
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        # get the max ct time file, essentially, the most recent file
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print(f"Restoring from {latest_checkpoint}")
        return keras.models.load_model(latest_checkpoint)
    print(f"Creating a new model")
    return get_compiled_model()

class TerminateOnNaN(tf.keras.callbacks.Callback):
    """
    Class to inherit from tf.keras.callback.Callback

    This terminates training on detecting NaN values in loss
    """
    def on_batch_end(self, batch, logs=None):
        if logs is not None and logs.get('loss', None) is not None and tf.math.is_nan(logs['loss']):
            print(f"\n!!!!NaN loss detected, terminating training!!!!\n")
            self.model.stop_training = True

def run_training(epochs=1):
    """
    Function to run training

    @param epochs int the number of epochs to run training for

    @return None
    """
    # instantiate a MirroredStrategy object
    # (optionally configuring which specific devices to use)
    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of devices: {strategy.num_replicas_in_sync}")

    # prepare a directory to sore all the checkpoints.
    checkpoint_dir = "./ckpt"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # use strategy object to open a scope, and within this scope,
    # create all Keras objects I need that contain variables
    # here, I create and compile the model
    # Open a strategy scope.
    with strategy.scope():
        # Everything that creates variables should be under the strategy scope.
        # In general this is only model construction & compile().
        model = make_or_restore_model(checkpoint_dir)

    # create a few callbacks to call when training the model
    callbacks = [
        # This callback saves a SavedModel every 20 steps
        # The epoch name is included in the folder name
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir + "/ckpt-{epoch:02d}-{batch:02d}.keras",
            save_freq=50,
        ),
        TerminateOnNaN(),
    ]

    # Train the model on all available devices.
    train_dataset, val_dataset, test_dataset, bs = get_dataset()
    print(f"Batch size is {bs}")
    model.fit(
        train_dataset,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_dataset,
    )

    # Test the model on all available devices.
    model.evaluate(test_dataset)


def main():
    # Running the first time creates the model
    run_training(epochs=2)


if __name__ == '__main__':
    main()
