from calendar import different_locale
import difflib
import shutil
from sklearn.metrics import roc_curve, auc, r2_score
from sympy import difference_delta
import uproot3 as up3
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
import numpy as np
import uproot as up
from configparser import ConfigParser
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
import pandas as pd
import time
import os
import shutil
import logging
from sklearn.metrics import roc_auc_score
import sys
from numpy import savez_compressed
from numpy import load
from sklearn.inspection import permutation_importance

pd.options.mode.chained_assignment = None

# create logger
logger = logging.getLogger('runPrediction.Prediction')

class Prediction(object):
    '''
    Class implementing a prediction script for one NN model for photon ID
    '''
    def __init__(self, file):
        self.config = ConfigParser()
        self.config.read(file)

        # prediction variables
        section = "prediction"
        self.data_files = self.config[section]['data_files'].split(',')
        self.folder = self.config[section]['folder']
        self.network_type = self.config[section]['network_type']
        self.conversion_type = self.config[section]['conversion_type']
        self.configfile = self.config[section]['configname']


        logger.info('Finished loading all config variables')


    def loadData(self):
        """ 
        This function loads in the specified data file of the specified conversion type for prediction as well as the specified scaler and network model
        """
        list_of_branches_ShowerShapes = ['y_Reta', 'y_Rphi', 'y_weta1', 'y_weta2', 'y_deltae', 'y_fracs1', 'y_Eratio', 'y_wtots1', 'y_Rhad', 'y_Rhad1', 'y_f1', 'y_e277']
        list_of_branches_Isolations = ['y_ptcone20', 'y_ptcone40', 'y_topoetcone20', 'y_topoetcone40']
        list_of_branches_for_binning = ['y_eta', 'y_pt', 'evt_mu', 'y_convType']
        list_of_branches_for_selection = ['y_isTruthMatchedPhoton', 'acceptEventPtBin','y_IsLoose','y_IsTight', 'dataWeightPtBin']
        list_of_branches_data = list_of_branches_for_binning + list_of_branches_ShowerShapes + list_of_branches_Isolations + list_of_branches_for_selection + ['weight']

        logger.info(f'Loading {self.data_file}_{self.conversion_type}.npz dataset')
        start_time = time.time()
        
        array = load(os.path.join('Check', f'/cephfs/user/s6flkirf/master_thesis/data/{self.data_file}.npz'),  allow_pickle=True)
        array = array['arr_0']
        self.data = pd.DataFrame(array, columns=list_of_branches_data)

        logger.info(f"Loaded in {len(self.data.index)} data events in {np.round(time.time() - start_time,1)} seconds.")
        
        logger.info(f'Loading in scaler and saved neural network model out of the following directoy: {self.network_type}/plots/{self.folder}')
        self.saved_scaler = joblib.load(f"/cephfs/user/s6herose/Bachelorarbeit/Programmieren/codemasterthesis/finalScripts/{self.network_type}/plots/{self.folder}/scaler.gz")
        print('now the model')
        self.saved_model = tf.keras.models.load_model(f'/cephfs/user/s6herose/Bachelorarbeit/Programmieren/codemasterthesis/finalScripts/{self.network_type}/plots/{self.folder}/model/classifier_model', compile=False)
        self.saved_model.summary()


    def loadDataimportance(self):       

        """ 
        This function loads in the specified data file of the specified conversion type for prediction as well as the specified scaler and network model
        """
        list_of_branches_ShowerShapes = ['y_Reta', 'y_Rphi', 'y_weta1', 'y_weta2', 'y_deltae', 'y_fracs1', 'y_Eratio', 'y_wtots1', 'y_Rhad', 'y_Rhad1', 'y_f1', 'y_e277']
        list_of_branches_Isolations = ['y_ptcone20', 'y_ptcone40', 'y_topoetcone20', 'y_topoetcone40']
        list_of_branches_for_binning = ['y_eta', 'y_pt', 'evt_mu', 'y_convType']
        list_of_branches_for_selection = ['y_isTruthMatchedPhoton', 'acceptEventPtBin','y_IsLoose','y_IsTight', 'dataWeightPtBin']
        list_of_branches_data = list_of_branches_for_binning + list_of_branches_ShowerShapes + list_of_branches_Isolations + list_of_branches_for_selection #+ ['weight']

        logger.info(f'Loading {self.data_file}_{self.conversion_type}.npz dataset')
        start_time = time.time()
        
        array = load(os.path.join('Check', f'/cephfs/user/s6flkirf/master_thesis/data/{self.data_file}.npz'),  allow_pickle=True)
        array = array['arr_0']
        self.data = pd.DataFrame(array, columns=list_of_branches_data)

        logger.info(f"Loaded in {len(self.data.index)} data events in {np.round(time.time() - start_time,1)} seconds.")

        
        array = load(os.path.join('Check', f'/cephfs/user/s6flkirf/master_thesis/data/sgn_{self.conversion_type}_FINAL.npz'),  allow_pickle=True)
        array = array['arr_0']
        self.sgn = pd.DataFrame(array, columns=list_of_branches_data)

        print('Row count signal is:',len(self.sgn.index))
        print('Row count background is:',len(self.data.index))



        # adding labels for classification
        self.sgn['class'] = 1 
        self.data['class'] = 0

        
        # shuffle dataframes
        self.sgn = shuffle(self.sgn)
        self.data = shuffle(self.data)
        sgn = self.sgn.to_numpy()
        bkg = self.data.to_numpy()
        
        # merge signal and background
        xdata = np.zeros((self.sgn.shape[0]+self.data.shape[0],self.sgn.shape[1]))
        lentgh_sgn = len(sgn)
        lentgh_bkg = len(bkg)
        for idx in range(0, lentgh_sgn):
            xdata[idx] = sgn[idx]
        for idx in range(0, lentgh_bkg):
            xdata[idx+lentgh_sgn] = bkg[idx]
        self.data = pd.DataFrame(xdata, columns=self.sgn.columns.values.tolist())
        
        
        self.y_true = self.data['class']
        self.w = self.data['weight']
        self.y_true = self.y_true.to_numpy()
        self.w = self.w.to_numpy()

        logger.info(f'Loading in scaler and saved neural network model out of the following directoy: {self.network_type}/plots/{self.folder}')
        self.saved_scaler = joblib.load(f"/cephfs/user/s6herose/Bachelorarbeit/Programmieren/codemasterthesis/finalScripts/{self.network_type}/plots/{self.folder}/scaler.gz")
        self.saved_model = tf.keras.models.load_model(f'/cephfs/user/s6herose/Bachelorarbeit/Programmieren/codemasterthesis/finalScripts/{self.network_type}/plots/{self.folder}/model/classifier_model')
        self.saved_model.summary()

        

    def predictData(self):
        """
        This function applies the scalar and predicts with the model (it loads the list of training variables out of the network_type/plots/folder/config.ini)
        """
        # transform data and predict
        self.config_network = ConfigParser()
        self.config_network.read(f"/cephfs/user/s6herose/Bachelorarbeit/Programmieren/codemasterthesis/finalScripts/{self.network_type}/plots/{self.folder}/{self.configfile}.ini")
        section = "general"
        self.training_variables = self.config_network[section]['training_variables'].split(',')

        logger.info(f"The following training variables have been found in the {self.network_type}/plots/{self.folder}/config.ini file and will be used for the scaler transformation: \n {self.training_variables}")

        # save root variables before applying scaler
        self.branches_root = ['y_pt', 'y_eta', 'y_isTruthMatchedPhoton', 'y_ptcone20', 'y_ptcone40', 'y_topoetcone20', 'y_topoetcone40', 'y_IsLoose', 'y_IsTight', 'y_convType', 'dataWeightPtBin', 'evt_mu'] 
        self.root_save = self.data[self.branches_root]
        
        self.data[self.training_variables] = self.saved_scaler.transform(self.data[self.training_variables])
        
        self.y_pred = self.saved_model.predict(self.data[self.training_variables], verbose=1)
    
    def importancevariable(self):
        """
        This function uses the classifier and the training data, to determine how big the importance of each variable for the classifier is.
        """
        #transform the data
        self.config_network = ConfigParser()
        self.config_network.read(f"/cephfs/user/s6herose/Bachelorarbeit/Programmieren/codemasterthesis/finalScripts/{self.network_type}/plots/{self.folder}/config.ini")
        section = "general"
        self.training_variables = self.config_network[section]['training_variables'].split(',')

        logger.info(f"The following training variables have been found in the {self.network_type}/plots/{self.folder}/config.ini file and will be used for the scaler transformation: \n {self.training_variables}")
        
        
        #calculate the importance
        self.data[self.training_variables] = self.saved_scaler.transform(self.data[self.training_variables])
        self.y_pred = self.saved_model.predict(self.data[self.training_variables], verbose=1)
        
        importance = np.ones((5,14))

        
        fpr, tpr, threshold = roc_curve(self.y_true, self.y_pred, sample_weight=self.w)

        a=np.column_stack([fpr, tpr])
        ind=np.argsort(a[:,0])
        fpr=a[ind][:,0]
        tpr=a[ind][:,1]

        auct = auc(fpr, tpr)
        print(auct)
        
        shuffle = self.data[self.training_variables[i]].sample(frac = 1)
        shuffledataframe = self.data
        shuffledataframe = shuffledataframe.drop(self.training_variables[i],1)
        shuffledataframe.insert(i, self.training_variables[i],shuffle.to_numpy())
        self.fake_y_pred = self.saved_model.predict(shuffledataframe[self.training_variables], verbose=1)
                
        self.importance = pd.DataFrame(importance, columns=self.training_variables)
        means=self.importance.mean()
        std = self.importance.std()
        self.meanstd = pd.concat([means, std.reindex(means.index)], axis=1)
        print(self.meanstd)




    def saveToRoot(self):
        """
        This function adds the y_pred (network predictions column) as well as a mcTotWeight dummy column set to 1 and saves the root file into the network_type/plots/folder/ directory where the MC predicitons are
        """
        self.root_save['y_pred'] = self.y_pred[0]
        self.root_save['mcTotWeight'] = 1
        try:
            os.remove(f"/cephfs/user/s6herose/Bachelorarbeit/Programmieren/codemasterthesis/finalScripts/{self.network_type}/plots/{self.folder}/{self.data_file}_{self.conversion_type}_NN_results.root")
        except OSError:
            pass

        # create root file fro data
        data_root = up3.recreate(f"/cephfs/user/s6herose/Bachelorarbeit/Programmieren/codemasterthesis/finalScripts/{self.network_type}/plots/{self.folder}/{self.data_file}_{self.conversion_type}_NN_results.root", compression=up3.ZLIB(4))
        data_root["SinglePhoton"] = up3.newtree({"y_pt": np.float64 , "y_eta": np.float64 , "y_isTruthMatchedPhoton": np.float64 , "y_ptcone20": np.float64 ,
                                            "y_ptcone40": np.float64 , "y_topoetcone20": np.float64 , "y_topoetcone40": np.float64 , "y_IsLoose": np.float64 , "y_IsTight": np.float64 , "y_convType": np.float64 ,
                                            "dataWeightPtBin": np.float64 , "evt_mu": np.float64 , "y_pred": np.float64 , "mcTotWeight": np.float64})
        data_root["SinglePhoton"].extend({"y_pt": self.root_save["y_pt"] , "y_eta": self.root_save["y_eta"] , "y_isTruthMatchedPhoton": self.root_save["y_isTruthMatchedPhoton"] , "y_ptcone20": self.root_save["y_ptcone20"] ,
                                            "y_ptcone40": self.root_save["y_ptcone40"] , "y_topoetcone20": self.root_save["y_topoetcone20"] , "y_topoetcone40": self.root_save["y_topoetcone40"] , "y_IsLoose": self.root_save["y_IsLoose"] , "y_IsTight": self.root_save["y_IsTight"] , "y_convType": self.root_save["y_convType"] ,
                                            "dataWeightPtBin": self.root_save["dataWeightPtBin"] , "evt_mu": self.root_save["evt_mu"] , "y_pred": self.root_save["y_pred"] , "mcTotWeight": self.root_save["mcTotWeight"]})

    def saveTonumpy(self):
        
        #saving the prediction
        self.root_save['y_pred'] = self.y_pred
        savez_compressed(os.path.join('Check', f"/cephfs/user/s6herose/Bachelorarbeit/Programmieren/codemasterthesis/finalScripts/{self.network_type}/plots/{self.folder}/{self.data_file}_NN_results.npz"), self.root_save.to_numpy())

        #saving the variable importance
        #savez_compressed(os.path.join('Check', f"/cephfs/user/s6herose/Bachelorarbeit/Programmieren/codemasterthesis/finalScripts/{self.network_type}/plots/{self.folder}/importance.npz"), self.meanstd.to_numpy())


    def runPrediction(self):
        """
        This function runs the predcition for all data files specified in the config variable self.data_files
        """
        logger.info(f"Starting the prediction loop for the following datasets {self.data_files} for {self.conversion_type} photons")
        for key in self.data_files:
            logger.info(f"Executing the prediction for the {key} dataset for {self.conversion_type} photons")
            self.data_file = key
            self.loadData()
            #self.loadDataimportance()
            self.predictData()
            #self.importancevariable()
            #self.saveTonumpy()
            self.saveToRoot()