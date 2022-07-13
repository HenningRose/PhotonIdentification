import configparser
import os
import sys
import logging
import inspect
import numpy as np
import pandas as pd
import prediction as prediction
import tensorflow as tf


logger = logging.getLogger('runPrediction')
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s:  %(message)s')
logger.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Finished loading all packages')
logger.info(f"Number of GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")


# prediction initialization
prediction = prediction.Prediction(sys.argv[1])
for key in prediction.__dict__:
    if (type(prediction.__dict__[key]) is np.ndarray):
        logger.info('Variable {} has been set to numpy array of shape{}'.format(key, prediction.__dict__[key].shape))
    else:
        logger.info('Variable {} has been set to {}'.format(key, prediction.__dict__[key]))

prediction.runPrediction()
