
# Train the recommender model using a Gradient Workflow
#
# This script is minimally commented because it is called from and explained by
# the recommender project notebook, deep_learning_recommender_tf.ipynb
#
# Code is the minimal subset of the notebook to perform the steps
#
# Lines are the same as the notebook, except where they need to be changed to
# work in a .py script as opposed to a .ipynb notebook
#
# Last updated: Aug 03rd 2021

# Setup

import subprocess

subprocess.run('pip install --upgrade pip', shell=True, check=True, stdout=subprocess.PIPE, universal_newlines=True)
subprocess.run('pip install -q tensorflow-recommenders==0.4.0', shell=True, check=True, stdout=subprocess.PIPE, universal_newlines=True)
subprocess.run('pip install -q --upgrade tensorflow-datasets==4.2.0', shell=True, check=True, stdout=subprocess.PIPE, universal_newlines=True)

import os
import platform
import pprint
import tempfile

from typing import Dict, Text

import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

# Get model hyperparameters

hp_final_epochs = int(os.environ.get('HP_FINAL_EPOCHS'))
hp_final_lr = float(os.environ.get('HP_FINAL_LR'))

# Data preparation

ratings_raw = tfds.load("movielens/100k-ratings", split="train")

ratings = ratings_raw.map(lambda x: {
    "movie_title": x["movie_title"],
    "timestamp": x["timestamp"],
    "user_id": x["user_id"],
    "user_rating": x["user_rating"]
})

timestamps = np.concatenate(list(ratings.map(lambda x: x["timestamp"]).batch(100)))

max_time = timestamps.max()
min_time = timestamps.min()

sixtieth_percentile = min_time + 0.6*(max_time - min_time)
eightieth_percentile = min_time + 0.8*(max_time - min_time)

train =      ratings.filter(lambda x: x["timestamp"] <= sixtieth_percentile)
validation = ratings.filter(lambda x: x["timestamp"] > sixtieth_percentile and x["timestamp"] <= eightieth_percentile)
test =       ratings.filter(lambda x: x["timestamp"] > eightieth_percentile)

ntimes_tr = 0
ntimes_va = 0
ntimes_te = 0

for x in train.take(-1).as_numpy_iterator():
  ntimes_tr += 1

for x in validation.take(-1).as_numpy_iterator():
  ntimes_va += 1

for x in test.take(-1).as_numpy_iterator():
  ntimes_te += 1

print("Number of rows in training set = {}".format(ntimes_tr))
print("Number of rows in validation set = {}".format(ntimes_va))
print("Number of rows in testing set = {}".format(ntimes_te))
print("Total number of rows = {}".format(ntimes_tr+ntimes_va+ntimes_te))

train = train.shuffle(ntimes_tr)
validation = validation.shuffle(ntimes_va)
test = test.shuffle(ntimes_te)

movie_titles = ratings.batch(1_000_000).map(lambda x: x["movie_title"])