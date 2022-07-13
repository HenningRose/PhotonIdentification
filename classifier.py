import numpy as np
import pandas as pd
import logging
from configparser import ConfigParser
import os
import sys
import plotClassifier as plot
import saveResults as save
import time
import dcor
from scipy.spatial import distance
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import metrics
import tensorflow.keras.backend as K
import kerastuner as kt

"""
This script stores all the functions that are needed to let the classifier run and train
"""


# create logger
logger = logging.getLogger('runClassifier.Classifier')

class Classifier(object):
    '''
    Class implementing an Classifier used for photon ID
    '''
    def __init__(self, file, train_dict, test_dict, root_variables):
        self.config = ConfigParser()
        self.config.read(file)

        # GENERAL variables
        section = "general"
        self.save_directory = self.config[section]['save_directory']
        self.seed = int(self.config[section]['seed'])
        self.verbose = int(self.config[section]['verbose'])
        self.save_model = self.config[section].getboolean('save_model')
        self.conversion_type = self.config[section]['conversion_type']
        self.plot_type  = int(self.config[section]['plot_type'])
        self.training_mode = self.config[section]['training_mode']
        np.random.seed(self.seed)

        # classifier variables
        section = "classifier"
        self.classifier_input_dim = train_dict['x_train'].shape[1]
        self.classifier_epochs = int(self.config[section]['classifier_epochs'])
        self.classifier_batch_size = int(self.config[section]['classifier_batch_size'])
        self.classifier_early_stopping = self.config[section].getboolean('classifier_early_stopping')
        if (self.training_mode == "normal"):
            self.classifier_use_dropout = self.config[section].getboolean('classifier_use_dropout')
            self.classifier_use_batch_norm = self.config[section].getboolean('classifier_use_batch_norm')
            layerList = self.config[section]['classifier_layers'].split(',')
            dropoutList = self.config[section]['classifier_dropout'].split(',')
            self.classifier_layers = [int(i) for i in layerList]
            self.classifier_dropout = [float(i) for i in dropoutList]
            self.classifier_learning_rate = float(self.config[section]['classifier_learning_rate'])
            self.classifier_optimizer_type = self.config[section]['classifier_optimizer_type']
            if (self.classifier_optimizer_type == "Adam"):
                self.classifier_optimizer = Adam(learning_rate = self.classifier_learning_rate)
            elif (self.classifier_optimizer_type == "SGD"):
                self.classifier_optimizer = SGD(learning_rate = self.classifier_learning_rate)
            else:
                sys.exit(f"The optimizer type {self.classifier_optimizer_type} is not available. Use Adam or SGD!")

        # optimizer variables
        section = "optimize"
        if (self.training_mode == "optimize"):
            self.num_steps = int(self.config[section]['num_steps'])
            num_nodes = self.config[section]['num_nodes'].split(',')
            self.num_nodes = [int(i) for i in num_nodes]
            num_layers = self.config[section]['num_layers'].split(',')
            self.num_layers = [int(i) for i in num_layers]
            dropout = self.config[section]['dropout'].split(',')
            self.dropout = [float(i) for i in dropout]
            leraning_rate = self.config[section]['leraning_rate'].split(',')
            self.leraning_rate = [float(i) for i in leraning_rate]

        # save variables
        section = "save"
        self.save_to_root = self.config[section].getboolean('save_to_root')
        self.save_to_numpy = self.config[section].getboolean('save_to_numpy')

        # DATA variables
        self.x_train_classifier = train_dict['x_train']
        self.y_train_classifier = train_dict['y_train']
        self.w_train_classifier = train_dict['w_train']
        self.w_original_train = train_dict['w_original_train']
        self.root_train = train_dict['root_save_train']
        self.x_test_classifier = test_dict['x_test']
        self.y_test_classifier = test_dict['y_test']
        self.w_test_classifier = test_dict['w_test']
        self.w_original_test = test_dict['w_original_test']
        self.root_test = test_dict['root_save_test']
        self.root_variables = root_variables
        logger.info('Finished loading all config variables')

###########################################
########### INITIALIZE NETWORKS ###########
###########################################
    
    def initializeClassifier(self):
        """
        Initalize the classifier with the config defined settings
        """
        logger.info('Building classifier')
        l1reg=1.e-8
        l2reg=1.e-2
        self.network_input = layers.Input(shape = (self.classifier_input_dim))
        # add dense, batch norm and dropout in for loop
        for i in range(len(self.classifier_layers)):
            if (i == 0):
                self.layer_classifier = layers.Dense(self.classifier_layers[i], activation = 'relu', kernel_regularizer=regularizers.l1_l2(l1=l1reg, l2=l2reg))(self.network_input)
                if (self.classifier_use_batch_norm):
                    self.layer_classifier = layers.BatchNormalization()(self.layer_classifier)
                if (self.classifier_use_dropout):
                    self.layer_classifier = layers.Dropout(self.classifier_dropout[i])(self.layer_classifier)
            else:
                self.layer_classifier = layers.Dense(self.classifier_layers[i], activation = 'relu', kernel_regularizer=regularizers.l1_l2(l1=l1reg, l2=l2reg))(self.layer_classifier)
                if (self.classifier_use_batch_norm):
                    self.layer_classifier = layers.BatchNormalization()(self.layer_classifier)
                if (self.classifier_use_dropout):
                    self.layer_classifier = layers.Dropout(self.classifier_dropout[i])(self.layer_classifier)
        self.layer_classifier_out = layers.Dense(1, activation = "sigmoid", name="output")(self.layer_classifier)
 
        self.model_classifier = Model(inputs = [self.network_input], outputs = [self.layer_classifier_out], name="classifier_training")
        self.model_classifier.compile(loss='binary_crossentropy', optimizer = self.classifier_optimizer)
       
        logger.info('Classifier built with the following structure:')
        self.model_classifier.summary()

    




###########################################
############ TRAINING FUNCTION ############
###########################################

    def trainClassifier(self):
        '''
        Training the classifier model.
        '''
        self.callbacks = None
        # early stopping callback
        if (self.classifier_early_stopping):
            self.callbacks = EarlyStopping(monitor='val_loss', mode="min", patience=5)


        logger.info(f'Training classifier with {self.classifier_epochs} epochs.')
        self.classifier_history = self.model_classifier.fit(  x = self.x_train_classifier, y = self.y_train_classifier, sample_weight=self.w_train_classifier,
                                                                    validation_data = (self.x_test_classifier, self.y_test_classifier, self.w_test_classifier),
                                                                    epochs=self.classifier_epochs, batch_size = self.classifier_batch_size, 
                                                                    callbacks = self.callbacks ,verbose = self.verbose
                                                                    )
                                                                       
        logger.info("Finished training the classifier")
    

    def trainOptimizeClassifier(self):
        '''
        Run the parameter optimization.
        '''
        def initalizeOptimizeClassifier(hp):
            """
            Initalize the classifier with the config defined settings for bayesian optimization
            """
            
            hp_nodes = hp.Int('nodes', min_value=self.num_nodes[0], max_value=self.num_nodes[1], step=self.num_nodes[2])
            hp_layers = hp.Int('layers', min_value=self.num_layers[0], max_value=self.num_layers[1], step=self.num_layers[2])
            hp_use_dropout = hp.Boolean('use_dropout', default=False)
            hp_dropout = hp.Float('dropout', min_value=self.dropout[0], max_value=self.dropout[1], step=self.dropout[2])
            hp_use_batch_norm = hp.Boolean('use_batch_norm', default=False)
            hp_optimizer = hp.Choice("optimizer", values=["Adam", "SGD"], default="Adam")
            hp_learning_rate = hp.Choice("learning_rate", values=self.leraning_rate, default=0.001)
    
            if (hp_optimizer == "Adam"):
                classifier_optimizer = Adam(learning_rate = hp_learning_rate)
            elif (hp_optimizer == "SGD"):
                classifier_optimizer = SGD(learning_rate = hp_learning_rate)
            logger.info('Building classifier for bayesian optimization')
            l1reg=1.e-8
            l2reg=1.e-2
            network_input = layers.Input(shape = (self.classifier_input_dim,))
            # add dense, batch norm and dropout in for loop
            for i in range(0, hp_layers):
                if (i == 0):
                    layer_classifier = layers.Dense(hp_nodes, activation = 'relu', kernel_regularizer=regularizers.l1_l2(l1=l1reg, l2=l2reg))(network_input)
                    if (hp_use_batch_norm):
                        layer_classifier = layers.BatchNormalization()(layer_classifier)
                    if (hp_use_dropout):
                        layer_classifier = layers.Dropout(hp_dropout)(layer_classifier)
                else:
                    layer_classifier = layers.Dense(hp_nodes, activation = 'relu', kernel_regularizer=regularizers.l1_l2(l1=l1reg, l2=l2reg))(layer_classifier)
                    if (hp_use_batch_norm):
                        layer_classifier = layers.BatchNormalization()(layer_classifier)
                    if (hp_use_dropout):
                        layer_classifier = layers.Dropout(hp_dropout)(layer_classifier)
            layer_classifier_out = layers.Dense(1, activation = "sigmoid", name="output")(layer_classifier)
    
            model_classifier = Model(inputs = [network_input], outputs = [layer_classifier_out], name="classifier_training")
            model_classifier.compile(loss='binary_crossentropy', optimizer = classifier_optimizer)
        
            logger.info('Classifier built with the following structure:')
            model_classifier.summary()
            return model_classifier

        self.callbacks = None
        # early stopping callback
        if (self.classifier_early_stopping):
            self.callbacks = EarlyStopping(monitor='val_loss', mode="min", patience=5)

        tuner = kt.BayesianOptimization( initalizeOptimizeClassifier,
                                         objective=kt.Objective('val_loss', direction="min"),
                                         num_initial_points=5,
                                         max_trials=self.num_steps,
                                         directory=f'/cephfs/user/s6herose/Bachelorarbeit/Programmieren/codemasterthesis/finalScripts/classifier/plots/{self.save_directory}/',
                                         project_name='bayesOpt')
                                            
        tuner.search(   x=self.x_train_classifier, y=self.y_train_classifier, sample_weight=self.w_train_classifier, 
                        validation_data = (self.x_test_classifier, self.y_test_classifier, self.w_test_classifier),
                        epochs=self.classifier_epochs, batch_size = self.classifier_batch_size, 
                        callbacks = [self.callbacks] ,verbose = self.verbose
                        )
        # Get the optimal hyperparameters
        best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

        logger.info(f"""The hyperparameter search is complete.
                The best network setup is:
                Number of Layers: {best_hps.get('layers')} \n
                Nodes per layer: {best_hps.get('nodes')} \n
                Use dropout: {best_hps.get('use_dropout')} \n
                Dropout percentage: {best_hps.get('dropout')} \n
                Use batch normalization: {best_hps.get('use_batch_norm')} \n
                Optimizer: {best_hps.get('optimizer')} \n
                Optimizer learning rate: {best_hps.get('learning_rate')} 
                """)

        # save values to txt file
        text_file = open(f"/cephfs/user/s6herose/Bachelorarbeit/Programmieren/codemasterthesis/finalScripts/classifier/plots/{self.save_directory}/hyperparameters.txt", "w")
        text_file.write(f"""The hyperparameter search is complete.
                        The best network setup is:
                        Number of Layers: {best_hps.get('layers')}
                        Nodes per layer: {best_hps.get('nodes')}
                        Use dropout: {best_hps.get('use_dropout')}
                        Dropout percentage: {best_hps.get('dropout')}
                        Use batch normalization: {best_hps.get('use_batch_norm')}
                        Optimizer: {best_hps.get('optimizer')} 
                        Optimizer learning rate: {best_hps.get('learning_rate')}
                        """)
        text_file.close()
        
        # Build the model with the optimal hyperparameters and train it
        self.model_classifier = tuner.hypermodel.build(best_hps)
        logger.info(f'Training classifier with {self.classifier_epochs} epochs.')

        self.classifier_history = self.model_classifier.fit(  x = self.x_train_classifier, y = self.y_train_classifier, sample_weight=self.w_train_classifier,
                                                                    validation_data = (self.x_test_classifier, self.y_test_classifier, self.w_test_classifier),
                                                                    epochs=self.classifier_epochs, batch_size = self.classifier_batch_size, 
                                                                    callbacks = self.callbacks ,verbose = self.verbose
                                                                    )
                                                                       
        logger.info("Finished training the classifier")        

 

###########################################
######### PLOTTING AND PREDICTION #########
###########################################

    def predModels(self):
        '''
        Predicting the classifier model.
        '''
        logger.info("Prediciting classifier output after training")
        # pred classifier
        self.y_train_classifier_pred = self.model_classifier.predict(self.x_train_classifier)
        self.y_test_classifier_pred = self.model_classifier.predict(self.x_test_classifier)
        self.root_train = pd.DataFrame(self.root_train, columns=self.root_variables)
        self.root_test = pd.DataFrame(self.root_test, columns=self.root_variables)
        self.isTight_train = self.root_train["y_IsTight"]
        self.isTight_test = self.root_test["y_IsTight"]
        self.iso_train = self.root_train["y_ptcone40"]
        self.iso_test = self.root_test["y_ptcone40"]
        

        if (self.save_model):
            if not os.path.exists('/cephfs/user/s6herose/Bachelorarbeit/Programmieren/codemasterthesis/finalScripts/classifier/plots/{}/model/'.format(self.save_directory)):
                os.makedirs('/cephfs/user/s6herose/Bachelorarbeit/Programmieren/codemasterthesis/finalScripts/classifier/plots/{}/model/'.format(self.save_directory))
            logger.info("Saving the model trained model in a sub directory")
            self.model_classifier.save('/cephfs/user/s6herose/Bachelorarbeit/Programmieren/codemasterthesis/finalScripts/classifier/plots/{}/model/classifier_model'.format(self.save_directory))



    def plotResults(self):
        '''
        Plot the results of the training
        '''
        logger.info("Plots are being created and saved in the /plots/{}/ sub-directory".format(self.save_directory))
        plot.Loss(self.conversion_type, self.classifier_history, self.save_directory)
        plot.classifierPredictions(self.conversion_type, self.y_train_classifier, self.y_train_classifier_pred, self.w_original_train, 
                                   self.y_test_classifier, self.y_test_classifier_pred, self.w_original_test, 
                                   self.plot_type, self.save_directory) 
        plot.ROC(self.conversion_type, self.y_train_classifier, self.y_train_classifier_pred, self.w_original_train,
                 self.y_test_classifier, self.y_test_classifier_pred, self.w_original_test,
                 self.plot_type, self.save_directory)  
        plot.Efficiency(self.conversion_type, self.y_train_classifier, self.y_train_classifier_pred, self.w_original_train, self.isTight_train, 
                        self.y_test_classifier, self.y_test_classifier_pred, self.w_original_test, self.isTight_test,
                        self.save_directory)
        plot.JSD(   self.conversion_type, self.y_train_classifier, self.y_train_classifier_pred, self.w_original_train, self.isTight_train, self.iso_train,
                    self.y_test_classifier, self.y_test_classifier_pred, self.w_original_test, self.isTight_test, self.iso_test,
                    self.plot_type, self.save_directory)
        logger.info("Plotting classifier output after training done")


###########################################
############# SAVING RESULTS ##############
###########################################

    def saveResults(self):
        if (self.save_to_numpy):
            logger.info("Saving the chosen variables to a compressed npz file (test and training seperately).")
            save.ToNumpy(   self.y_train_classifier_pred, self.w_original_train, self.root_train,
                            self.y_test_classifier_pred, self.w_original_test, self.root_test,
                            self.save_directory)
        if (self.save_to_root):
            logger.info("Saving the root_variables in a ROOT file (test and training merged).")
            save.ToRoot(    self.y_train_classifier_pred, self.w_original_train, self.root_train,
                            self.y_test_classifier_pred, self.w_original_test, self.root_test,
                            self.conversion_type, self.save_directory)