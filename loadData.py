import numpy as np
import uproot as up
from configparser import ConfigParser
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
import pandas as pd
import plotClassifier as plot
import time
import os
import shutil
import logging
import sys
from numpy import savez_compressed
from numpy import load
pd.options.mode.chained_assignment = None  # default='warn'

""" 
This script contains all functions that are used to load in and preprocess the PhotonID data
"""

# create logger
logger = logging.getLogger('runClassifier.loadInput')

class DataLoader(object):
    '''
    Class loading the input data for the photon ID classifier
    '''
    def __init__(self, file):
        self.config = ConfigParser()
        self.config.read(file)

        # GENERAL variables
        section = "general"
        self.save_directory = self.config[section]['save_directory']
        self.reweighting = self.config[section]['reweighting']
        self.entry_stop = int(self.config[section]['entry_stop'])
        self.training_variables = self.config[section]['training_variables'].split(',')
        self.conversion_type = self.config[section]['conversion_type']
        self.selection_type = self.config[section]['selection_type']
        selection_list = self.config[section]['selection_list'].split(',')
        self.selection_list = [float(i) for i in selection_list]
        self.scaler_type = self.config[section]['scaler_type']
        self.split_size = float(self.config[section]['split_size'])
        self.save_scaler = self.config[section].getboolean('save_scaler')
        self.plot_type  = int(self.config[section]['plot_type'])
        

        if not os.path.exists('/cephfs/user/s6herose/Bachelorarbeit/Programmieren/codemasterthesis/finalScripts/classifier/plots/{}'.format(self.save_directory)):
            os.makedirs('/cephfs/user/s6herose/Bachelorarbeit/Programmieren/codemasterthesis/finalScripts/classifier/plots/{}'.format(self.save_directory))
        
        newPath = shutil.copy(file, '/cephfs/user/s6herose/Bachelorarbeit/Programmieren/codemasterthesis/finalScripts/classifier/plots/{}'.format(self.save_directory))
        logger.info(f'Saved config file into the {self.save_directory} directory')



    def loadInput(self):
        """ 
        This function loads in the signal and background dataframes, shuffles, \n
        applys additional selection for pt and eta and specifys an input limit
        """

        if (self.conversion_type != "converted" and self.conversion_type != "unconverted"):
            sys.exit(f"The conversion type {self.conversion_type} is not available. Use converted or converted")

        # specify all branches that are needed
        list_of_branches_ShowerShapes = ['y_Reta', 'y_Rphi', 'y_weta1', 'y_weta2', 'y_deltae', 'y_fracs1', 'y_Eratio', 'y_wtots1', 'y_Rhad', 'y_Rhad1', 'y_f1', 'y_e277']
        list_of_branches_Isolations = ['y_ptcone20', 'y_ptcone40', 'y_topoetcone20', 'y_topoetcone40']
        list_of_branches_for_binning = ['y_eta', 'y_pt', 'evt_mu', 'y_convType']
        list_of_branches_for_selection = ['y_isTruthMatchedPhoton', 'acceptEventPtBin','y_IsLoose','y_IsTight', 'dataWeightPtBin']
        weight_branch=['mcTotWeight']
        list_of_branches = list_of_branches_for_binning + list_of_branches_ShowerShapes + list_of_branches_Isolations + list_of_branches_for_selection + weight_branch

        # load signal and background array
        logger.info(f'loadFiles: Loading {self.conversion_type} datasets')

        start_time = time.time()
        array = load(os.path.join('Check', f'/cephfs/user/s6flkirf/master_thesis/data/bkg_{self.conversion_type}_FINAL.npz'),  allow_pickle=True)
        array = array['arr_0']
        self.bkg_df = pd.DataFrame(array, columns=list_of_branches)
        logger.info(f"Loaded in {len(self.bkg_df.index)} background events in {np.round(time.time() - start_time,1)} seconds.")

        start_time = time.time()
        array = load(os.path.join('Check', f'/cephfs/user/s6flkirf/master_thesis/data/sgn_{self.conversion_type}_FINAL.npz'),  allow_pickle=True)
        array = array['arr_0']
        self.sgn_df = pd.DataFrame(array, columns=list_of_branches)
        logger.info(f"Loaded in {len(self.sgn_df.index)} signal events in {np.round(time.time() - start_time,1)} seconds.")

        # adding labels for classification
        self.sgn_df['class'] = 1 
        self.bkg_df['class'] = 0

        # shuffle dataframes
        self.sgn_df = shuffle(self.sgn_df)
        self.bkg_df = shuffle(self.bkg_df)

        # apply additional selection on pt and eta if desired
        if (self.selection_type == 'eta_cut'):
            self.sgn_df = self.sgn_df.loc[(abs(self.sgn_df['y_eta']) >= self.selection_list[2]) & (abs(self.sgn_df['y_eta']) <= self.selection_list[3])]
            self.bkg_df = self.bkg_df.loc[(abs(self.bkg_df['y_eta']) >= self.selection_list[2]) & (abs(self.bkg_df['y_eta']) <= self.selection_list[3])]
            logger.info('Training on {} < |eta| < {} dataset'.format(self.selection_list[2], self.selection_list[3]))
        elif (self.selection_type == 'pt_cut'):
            self.sgn_df = self.sgn_df.loc[(self.sgn_df['y_pt'] >= self.selection_list[0]) & (self.sgn_df['y_pt'] <= self.selection_list[1])]
            self.bkg_df = self.bkg_df.loc[(self.bkg_df['y_pt'] >= self.selection_list[0]) & (self.bkg_df['y_pt'] <= self.selection_list[1])]
            logger.info('Training on {} < pt < {} dataset'.format(self.selection_list[0], self.selection_list[1]))
        elif (self.selection_type == 'both_cut'):
            self.sgn_df = self.sgn_df.loc[(self.sgn_df['y_pt'] >= self.selection_list[0]) & (self.sgn_df['y_pt'] <= self.selection_list[1]) & (abs(self.sgn_df['y_eta']) >= self.selection_list[2]) & (abs(self.sgn_df['y_eta']) <= self.selection_list[3])]
            self.bkg_df = self.bkg_df.loc[(self.bkg_df['y_pt'] >= self.selection_list[0]) & (self.bkg_df['y_pt'] <= self.selection_list[1]) & (abs(self.bkg_df['y_eta']) >= self.selection_list[2]) & (abs(self.bkg_df['y_eta']) <= self.selection_list[3])]
            logger.info('Training on {} < pt < {} and {} < |eta| < {} dataset'.format(self.selection_list[0], self.selection_list[1], self.selection_list[2], self.selection_list[3]))
        elif (self.selection_type == 'normal'):
            logger.info('Training on unchanged dataset')
        else:
            sys.exit('ERROR: This cut option is not supported. Use eta_cut, pt_cut, both_cut or normal')

        logger.info(f"After cut: {len(self.bkg_df.index)} background events available.")
        logger.info(f"After cut: {len(self.sgn_df.index)} signal events available.")

        # drop rows above entry_stops
        if (len(self.sgn_df.index) > self.entry_stop):
            self.sgn_df = self.sgn_df.head(self.entry_stop)
        if (len(self.bkg_df.index) > self.entry_stop):
            self.bkg_df = self.bkg_df.head(self.entry_stop)
        logger.info(f"After row drop: {len(self.bkg_df.index)} background events available.")
        logger.info(f"After row drop: {len(self.sgn_df.index)} signal events available.")



    def sampleWeights(self):
        """
        This function creates new weights for the training \n
        Either normal which will just reweight so that the sum of background weight and signal weights is equal \n
        Or reweight which will reweight the background samples to match signal in all the pt and eta bins \n
        The new weights will be calles train_weight but mcTotWeight is unchanged
        """
        # create new weight variable and plot unchanged pt and eta
        self.sgn_df["train_weight"] = self.sgn_df["mcTotWeight"]
        self.bkg_df["train_weight"] = self.bkg_df["mcTotWeight"]
        plot.Distributions(self.conversion_type, self.sgn_df, self.bkg_df, "original", self.plot_type, self.save_directory)
        
        # do the reweighting and plot changed pt and eta
        if (self.reweighting == "normal"):
            logger.info("Using normal weights and just adjusting the sum to match for signal and background")
            self.bkg_df['train_weight'] = self.bkg_df['train_weight']*(self.sgn_df['train_weight'].sum()/self.bkg_df['train_weight'].sum())

        elif (self.reweighting == "reweight"):
            logger.info("Using reweighted weights in eta and pt bins so that signal maches background")
            def reweighting(original, target):   
                pt = [25,30,35,40,45,50,60,80,100,125,150,175,250,500,1500]
                eta = [0.0, 0.6, 0.8, 1.15, 1.37, 1.52, 1.81, 2.01, 2.37]

                changed_df = pd.DataFrame(columns=original.columns.values.tolist())
            
                for i in range(0, 8):
                    for j in range(0, 14):
                        original_df = original[((abs(original.y_eta) >= eta[i]) & (abs(original.y_eta) < eta[i+1])) & ((original.y_pt >= pt[j]) & (original.y_pt < pt[j+1]))]
                        target_df = target[((abs(target.y_eta) >= eta[i]) & (abs(target.y_eta) < eta[i+1])) & ((target.y_pt >= pt[j]) & (target.y_pt < pt[j+1]))]
                        if (original_df.empty):
                            continue
                        else:
                            original_sum = original_df['train_weight'].sum()
                            target_sum = target_df['train_weight'].sum()
                            original_df.train_weight = original_df.train_weight*(target_sum/original_sum)
                            changed_df = changed_df.append(original_df)
            
                changed_df = changed_df.sort_index()
                return changed_df
            self.bkg_df = reweighting(self.bkg_df, self.sgn_df)
        else:
            sys.exit('ERROR: This reweighting option is not supported. Use normal or reweight')
        plot.Distributions(self.conversion_type, self.sgn_df, self.bkg_df, "reweighted", self.plot_type, self.save_directory)

        logger.info("Reweighting of train_weight finished! Original weights mcTotWeights are still unchanged.")
        logger.info(f"Sum of train_weight for background events: {self.bkg_df['train_weight'].sum()}")
        logger.info(f"Sum of train_weight for signal events: {self.sgn_df['train_weight'].sum()}")

    def dataPreprocessing(self):
        """
        This function finishes the preprocessing by merging signal and background dataframes \n
        Both are shuffled before splitting into training and test sample \n
        A scaler (MinMax or Standard) is applied to the dataframes \n
        Finally dictionaries with all inputs, weights, targets and data for later ROOT files is saved
        """

        logger.info('Number of signal events loaded: {}'.format(self.sgn_df.shape[0]))
        logger.info('Number of background events loaded: {}'.format(self.bkg_df.shape[0]))
        logger.info('Scaler used in preprocessing: {}'.format(self.scaler_type))
        sgn = self.sgn_df.to_numpy()
        bkg = self.bkg_df.to_numpy()

        # merge signal and background
        data = np.zeros((self.sgn_df.shape[0]+self.bkg_df.shape[0],self.sgn_df.shape[1]))
        lentgh_sgn = len(sgn)
        lentgh_bkg = len(bkg)
        for idx in range(0, lentgh_sgn):
            data[idx] = sgn[idx]
        for idx in range(0, lentgh_bkg):
            data[idx+lentgh_sgn] = bkg[idx]
        self.data = pd.DataFrame(data, columns=self.sgn_df.columns.values.tolist())
    
        logger.info('Number of events available for training and testing: {}'.format(self.data.shape[0]))
    
        # shuffle and split
        self.data = shuffle(self.data)
        self.train_data, self.test_data = train_test_split(self.data, test_size=self.split_size, shuffle=False)

        # define arrays for ROOT file 
        self.branches_root = ['class', 'y_pt', 'y_eta', 'y_isTruthMatchedPhoton', 'y_ptcone20', 'y_ptcone40', 'y_topoetcone20', 'y_topoetcone40', 'y_IsLoose', 'y_IsTight', 'y_convType', 'dataWeightPtBin', 'evt_mu','y_Reta', 'y_Rphi', 'y_weta1', 'y_weta2', 'y_deltae', 'y_fracs1', 'y_Eratio', 'y_wtots1', 'y_Rhad', 'y_Rhad1', 'y_f1', 'y_e277']
        root_save_train, root_save_test = self.train_data[self.branches_root], self.test_data[self.branches_root]
        logger.info(f'Split size: {self.split_size}')
        logger.info(f'Training events: {self.train_data.shape[0]}')
        logger.info(f'Test events: {self.test_data.shape[0]}')

        # apply scaler to training and test data
        self.scaler = StandardScaler()
        if (self.scaler_type == 'MinMax'):
            self.scaler = MinMaxScaler()
        elif ((self.scaler_type != 'MinMax') & (self.scaler_type !='Standard')):
            sys.exit("The scaler_type {} is not available. Use MinxMax or Standard".format(self.scaler_type))
    
        self.scaler.fit(self.train_data[self.training_variables])
        self.train_data[self.training_variables] = self.scaler.transform(self.train_data[self.training_variables])
        self.test_data[self.training_variables] = self.scaler.transform(self.test_data[self.training_variables])
    
        logger.info("Fitted {} scaler to input data and transformed the datasets".format(self.scaler_type))

        if (self.save_scaler):
            joblib.dump(self.scaler, f"/cephfs/user/s6herose/Bachelorarbeit/Programmieren/codemasterthesis/finalScripts/classifier/plots/{self.save_directory}/scaler.gz")
            logger.info("Saved scaler")
        
        # define input, weights, target 
        x_train, y_train, w_train, w_original_train = self.train_data[self.training_variables], self.train_data['class'], self.train_data['train_weight'], self.train_data['mcTotWeight']
        x_test, y_test, w_test, w_original_test = self.test_data[self.training_variables], self.test_data['class'], self.test_data['train_weight'], self.test_data['mcTotWeight']
        
        logger.info("Final input variables: {}".format(list(x_train.columns)))
        logger.info("The following variables will be saved in the ROOT output file: {}".format(list(root_save_train.columns)))
    
    
        self.train_dict = {
            "x_train": x_train.to_numpy(),
            "y_train": y_train.to_numpy(),
            "w_train": w_train.to_numpy(),
            "w_original_train": w_original_train.to_numpy(),
            "root_save_train": root_save_train.to_numpy()
        }
    
        self.test_dict = {
            "x_test": x_test.to_numpy(),
            "y_test": y_test.to_numpy(),
            "w_test": w_test.to_numpy(),
            "w_original_test": w_original_test.to_numpy(),
            "root_save_test": root_save_test.to_numpy()
        }



