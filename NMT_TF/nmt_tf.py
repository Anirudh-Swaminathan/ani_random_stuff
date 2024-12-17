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

