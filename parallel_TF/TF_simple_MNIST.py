#!/usr/bin/env python3
"""
Program to train a simple Neural Network on MNIST with Keras
"""
# native imports here
# 3rd party imports here
import tensorflow as tf
import tensorflow_datasets as tfds


def normalize_img(image, label):
    """
    Normalizes images: `uint8` -> `float32`

    @param image tf.uint8 image to normalize
    @param label tf label

    @return tuple (tf.float32 normalized image, label)
    """
    return tf.cast(image, tf.float32) / 255., label

def main():
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
    ds_train = ds_train.batch(128)

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
    ds_test = ds_test.batch(128)

    # Caching is done after batching because batches can be the same between epochs
    ds_test = ds_test.cache()

    # prefetch for improved performance
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    # Create and train the model
    # create model with 784 neurons in input layer,
    # 128 neurons in hidden layer with ReLU,
    # and 10 neurons for 10 classes in output layer which output logits, not probabilities
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    # compile model with Adam optimizer
    # Sparse Categorical cross entropy is because the true labels are integers,
    # not 1-hot encoded vectors.
    # From logits because the final output layer does not have softmax, but outputs logits
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    # run training on the model now
    model.fit(
        ds_train,
        epochs=10,
        validation_data=ds_test,
    )


if __name__ == "__main__":
    main()
