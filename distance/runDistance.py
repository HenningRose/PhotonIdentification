import configparser
import os
import sys

import logging
import inspect
import numpy as np
import pandas as pd
import loadData
import distance as distance
import tensorflow as tf



logger = logging.getLogger('runDistance')
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

# distance initialization
distance = distance.Distance(sys.argv[1], data.train_dict, data.test_dict, data.branches_root)
for key in distance.__dict__:
    if (type(distance.__dict__[key]) is np.ndarray):
        logger.info('Variable {} has been set to numpy array of shape{}'.format(key, distance.__dict__[key].shape))
    elif (isinstance(distance.__dict__[key], pd.Series)):
        logger.info('Variable {} has been set to pandas series of shape{}'.format(key, distance.__dict__[key].shape))
    elif (isinstance(distance.__dict__[key], pd.DataFrame)):
        logger.info('Variable {} has been set to pandas dataframe of shape{}'.format(key, distance.__dict__[key].shape))
    else:
        logger.info('Variable {} has been set to {}'.format(key, distance.__dict__[key]))


if (distance.training_mode == "optimize"):
    # distance optimization
    distance.trainOptimizeDistance()
elif (distance.training_mode == "normal"):
    # distance training
    distance.initializeDistance()
    if (distance.classifier_pre_training == True):
        distance.preTrainDistance()
        distance.prePredModels()
        distance.prePlotResults()
    distance.trainDistance()


# distance predicting and plotting
distance.predModels()
distance.plotResults()

# save outputs
distance.saveResults()
