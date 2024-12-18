#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Anirudh Swaminathan
# @Version : 1.0
# @Description : Neural Machine Translation using TensorFlow

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

# print tokenized sentences
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

