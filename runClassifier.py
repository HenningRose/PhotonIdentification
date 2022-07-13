import configparser
import os
import sys

import logging
import inspect
import numpy as np
import pandas as pd
import loadData
import classifier as classifier
import tensorflow as tf


# TODO in this order
# add weights to training (class weights for regreessor??)
# add valdiation in training loop for loss
# make loss and prediction plot work
# make ROC plot work
# create on GPU??

logger = logging.getLogger('runClassifier')
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s:  %(message)s')
logger.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Finished loading all packages')
logger.info(f"Number of GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

# load and preprocess input data
data = loadData.DataLoader(sys.argv[1])
data.loadInput()
data.sampleWeights()
data.dataPreprocessing()

# classifier initialization
classifier = classifier.Classifier(sys.argv[1], data.train_dict, data.test_dict, data.branches_root)
for key in classifier.__dict__:
    if (type(classifier.__dict__[key]) is np.ndarray):
        logger.info('Variable {} has been set to numpy array of shape{}'.format(key, classifier.__dict__[key].shape))
    else:
        logger.info('Variable {} has been set to {}'.format(key, classifier.__dict__[key]))


if (classifier.training_mode == "optimize"):
    # classifier optimization
    classifier.trainOptimizeClassifier()
elif (classifier.training_mode == "normal"):
    # classifier training
    classifier.initializeClassifier()
    classifier.trainClassifier()

# classifier predicting and plotting
classifier.predModels()
classifier.plotResults()

# save outputs
classifier.saveResults()
