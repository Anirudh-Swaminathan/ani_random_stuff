#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Anirudh Swaminathan
# @Version : 1.0
# @Description : Neural Machine Translation using TensorFlow
# @Tutorial Link : https://www.tensorflow.org/text/tutorials/transformer

# native imports here
import logging
import time

# 3rd party imports here
import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow as tf

import tensorflow_text

# Download the dataset
examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']

# print 3 examples
for pt_examples, en_examples in train_examples.batch(3).take(1).cache():
    print("> Examples in Portuguese:")
    for pt in pt_examples.numpy():
        print(pt.decode('utf-8'))
    print()

    print(f"> Examples in English:")
    for en in en_examples.numpy():
        print(en.decode('utf-8'))

# Set up the Tokenizer
# download, extract, and import the saved_model
model_name = "ted_hrlr_translate_pt_en_converter"
tf.keras.utils.get_file(
        f"{model_name}.zip",
        f"https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip",
        cache_dir='.', cache_subdir='', extract=True
    )
tokenizers = tf.saved_model.load(model_name)

tok_li = [item for item in dir(tokenizers.en) if not item.startswith('_')]
print(f"Tokenizer methods available: {tok_li}")

# print text examples sentences
print(f"> This is a batch of strings:")
for en in en_examples.numpy():
    print(en.decode('utf-8'))

# tokenize the examples
print(f"Tokenizing the examples")
encoded = tokenizers.en.tokenize(en_examples)
print(f"> This is a padded-batch of token IDs:")
for row in encoded.to_list():
    print(row)

# detokenize the tokenized examples
print(f"Detokenizing the tokenized examples now")
round_trip = tokenizers.en.detokenize(encoded)
print(f"> This is human-readable text:")
for line in round_trip.numpy():
    print(line.decode('utf-8'))

# lower level lookup method
print(f"Lower level lookup method converts from token-IDs to token text")
print(f"> This is the test split into tokens:")
tokens = tokenizers.en.lookup(encoded)
print(tokens)

# Distribution of tokens per example in the dataset
lengths = list()
for pt_examples, en_examples in train_examples.batch(1024).cache():
    pt_tokens = tokenizers.pt.tokenize(pt_examples)
    lengths.append(pt_tokens.row_lengths())

    en_tokens = tokenizers.en.tokenize(en_examples)
    lengths.append(en_tokens.row_lengths())
    print('.', end='', flush=True)

all_lengths = np.concatenate(lengths)

# plot a histogram of the tokens per example in the dataset
plt.hist(all_lengths, np.linspace(0, 500, 101))
plt.ylim(plt.ylim())
max_length = max(all_lengths)
plt.plot([max_length, max_length], plt.ylim())
plt.title(f"Maximum tokens per example: {max_length}")
plt.savefig("token_distribution_histogram.png")
plt.show()

# Data pipeline with tf.data
# Tokenizes into ragged batches. Trims to be no longer than MAX_TOKENS. Splits the target tokens into inputs and labels.
# Shifts labels by 1 step so that at each input location, the label is the ID of the next token.
# Converts the RaggedTensors to padded dense Tensors.
# returns an (inputs, labels) pair.
MAX_TOKENS = 128
def prepare_batch(pt, en):
    """
    Tokenize, trim, and split the examples into an (inputs, labels) pair
    Converts the RaggedTensors to padded dense Tensors.
    @Args:
    pt: Portuguese text
    en: English text

    @Returns:
    (pt_tokens, en_inputs), en_labels pair
    """
    # Output is ragged.
    pt_tokens = tokenizers.pt.tokenize(pt)

    # Trim to MAX_TOKENS
    pt_tokens = pt_tokens[:, :MAX_TOKENS]

    # Convert to 0-padded dense Tensor
    pt_tokens = pt_tokens.to_tensor()

    # Tokenize the English text
    en_tokens = tokenizers.en.tokenize(en)

    # Trim to MAX_TOKENS + 1 to allow for shifted by 1 labels
    en_tokens = en_tokens[:, :(MAX_TOKENS + 1)]

    # Drop the [END] tokens
    en_inputs = en_tokens[:, :-1].to_tensor()

    # Drop the [START] tokens
    en_labels = en_tokens[:, 1:].to_tensor()

    return (pt_tokens, en_inputs), en_labels

# Convert dataset of text examples into data of batches for training
# Tokenize the text. Filter out sequences that are too long. Tokenizer is much more efficient on large batches.
# Cache the dataset to memory to get a speedup while reading from it.
# Shuffle and denst_to_ragged_batch randomizes the order and assembles batches of examples.
# prefetch runs the dataset in parallel with the model to ensure data is available when needed.
BUFFER_SIZE = 20000
BATCH_SIZE = 64
def make_batches(ds):
    """
    Convert dataset of text examples into data of batches for training
    @Args:
    ds: dataset of text examples

    @Returns:
    batched
    """
    return (
        ds
        .cache()
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .map(prepare_batch, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

# Test the Dataset
# create training and validation set batches
train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)

# Resulting tf.data.Dataset objects are setup for training with Keras.
# Model.fit training expects (inputs, labels) pairs.
# inputs are pairs of tokenized Portuguese and English sentences (pt_tokens, en_inputs).
# labels are the English sentences, offset by 1 token (en_labels).
# Similar to text generation, except the model has additional input "context" (the Portuguese sequence) that the model is conditioned on
# Teacher Forcing: regardless of the model's prediction at each timestep, it gets the true value as input for the next timestep.
# Simple and efficient, because the model doesn't need to run sequentially.
# Outputs at the different sequence locations are computed in parallel
# Alternative: Just provide Portuguese text as input and let the model generate the English translation.
# Alternative is slower since time steps can't run in parallel
# Alternative is a harder task to learn, since the model can't get the end of a sentence correct until it gets the beginning correct.
# Alternative can give a more stable model because the model has to learn to correct its own errors during training.
for (pt, en), en_labels in train_batches.take(1).cache():
    print(f"Portuguese shape: {pt.shape}")
    print(f"English shape: {en.shape}")
    print(f"English label shape: {en_labels.shape}")
    break

print(f"Checking if shifted labels match the English text")
print(en[0][:10])
print(en_labels[0][:10])
