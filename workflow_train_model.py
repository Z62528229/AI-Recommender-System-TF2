
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