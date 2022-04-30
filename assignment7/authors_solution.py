# Samson Nguyen
# 1001496565
# CSE 4392 Assignment 7
import os
import random
import shutil

import numpy as np
from keras import layers
from keras.layers import TextVectorization
from tensorflow import keras


def learn_model(train_files):
    train_dir = '/'.join(train_files[0][0].split("/")[:-1])
    authors = []
    training_set = []
    for s in train_files:
        sentences = []
        ctr = 0
        # read every file for this author
        for file in s:
            f = open(file)
            # list every sentence in sentences
            sentences += f.read().split(".")
            f.close()

        # randomly select 3000 sentences of at least 40 or more chars for this author
        while ctr < 3000:
            idx = random.randint(0, len(sentences) - 2)
            sentence = sentences[idx]
            if len(sentence.split(" ")) >= 40:
                training_set.append((sentence, len(authors)))
            else:
                training_set.append((sentence + " " + sentences[idx + 1], len(authors)))
            ctr += 1

        # consider this author read
        authors.append(s[0].split("/")[-1].split("_")[0])

    # organize files into author directories for text_dataset_from_directory
    for files, category in zip(train_files, authors):
        os.makedirs(train_dir + "/" + category)
        for file in files:
            shutil.copy(file, train_dir + "/" + category + "/" + file.split("/")[-1])

    train_ds = keras.utils.text_dataset_from_directory(train_dir, batch_size=32)
    max_tokens = 20000

    # text_vectorization makes a bag of words given input string
    text_vectorization = TextVectorization(max_tokens=max_tokens, output_mode="multi_hot", ngrams=2)
    text_vectorization.adapt(train_ds.map(lambda x, y: x))

    # make training sets categorical by text_vectorization on the input strings
    # categorical_1gram_train_ds = train_ds.map(lambda x, y: (text_vectorization(x), y))
    categorical_train_set = list((text_vectorization(x), y) for x, y in training_set)

    # Sequential model to be fitted
    # SHOULD output 1 integer for label
    model = keras.Sequential([keras.Input(shape=(max_tokens,)),
                              # layers.Dense(8 * len(authors), activation="tanh"),
                              # layers.Dropout(0.2),
                              layers.Dense(2 * len(authors), activation="tanh"),
                              # layers.Dense(len(authors)),
                              layers.Dense(len(authors), activation="softmax")])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # callbacks = [keras.callbacks.ModelCheckpoint("categorical_1gram.keras", save_best_only=True)]
    # model.fit(categorical_1gram_train_ds.cache(), epochs=10, callbacks=callbacks)

    # convert training data and labels into np arrays for keras
    train_x = np.asarray(list(a for a, b in categorical_train_set)).astype('int32')
    train_y = np.asarray(list(b for a, b in categorical_train_set)).astype('int32')
    model.fit(train_x, train_y, epochs=10)

    # add text_vectorization at the beginning of the model so that it can be fed raw strings
    new_model = keras.Sequential([text_vectorization, model])
    new_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # model = keras.models.load_model("categorical_1gram.keras")

    # clean/remove organized directories
    for files, category in zip(train_files, authors):
        shutil.rmtree(train_dir + "/" + category)
        # os.removedirs(train_dir + "/" + category)

    return new_model
