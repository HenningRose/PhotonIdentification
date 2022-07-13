import numpy as np
import pandas as pd
from tensorflow.python.autograph.impl import conversion
import uproot3 as up
from numpy import savez_compressed
import os

"""
This script handles the saving of the results into root and compresses npz files
"""

def ToNumpy(y_pred_train, weight_train, root_train, y_pred_test, weight_test, root_test, save_directory):
    """ This function saves the following variables to a compressed npz file: 'class', 'y_pt', 'y_eta', 'y_IsTight', 'y_ptcone40', 'y_pred', 'mcTotWeight' \n
    Arguments:
    y_train_pred, y_test_pred: predicted targets for test and training sample
    w_train, w_test: weights for test and training sample
    root_train, root_test: dataframe with variables defined to save
    save_directory: directory to save plot in
    """
    
    npz_train = root_train[['class', 'y_pt', 'y_eta', 'y_IsTight', 'track']]
    npz_test = root_test[['class', 'y_pt', 'y_eta', 'y_IsTight', 'track']]

    npz_train['y_pred'] = y_pred_train
    npz_train['mcTotWeight'] = weight_train

    npz_test['y_pred'] = y_pred_test
    npz_test['mcTotWeight'] = weight_test


    savez_compressed(os.path.join('Check', f"/cephfs/user/s6herose/Bachelorarbeit/Programmieren/codemasterthesis/finalScripts/distance/plots/{save_directory}/npz_train.npz"), npz_train.to_numpy())
    savez_compressed(os.path.join('Check', f"/cephfs/user/s6herose/Bachelorarbeit/Programmieren/codemasterthesis/finalScripts/distance/plots/{save_directory}/npz_test.npz"), npz_test.to_numpy())


def ToRoot(y_pred_train, weight_train, root_train, y_pred_test, weight_test, root_test, conversion_type, save_directory):
    """ This function saves the following variables in one ROOT file for signal and one for background events: \n
    'class', 'y_pt', 'y_eta', 'y_isTruthMatchedPhoton', 'y_ptcone20', 'y_ptcone40', 'y_topoetcone20', 'y_IsLoose', 'y_IsTight', 'y_convType', 'dataWeightPtBin', 'y_pred', 'mcTotWeight' \n
    Arguments:
    y_train_pred, y_test_pred: predicted targets for test and training sample
    w_train, w_test: weights for test and training sample
    root_train, root_test: dataframe with variables defined to save
    conversion_type: converted or unconverted
    save_directory: directory to save plot in
    """

    root_train['y_pred'] = y_pred_train
    root_train['mcTotWeight'] = weight_train

    root_test['y_pred'] = y_pred_test
    root_test['mcTotWeight'] = weight_test

    # merge training and test
    columns = root_train.columns.values.tolist()
    root_train = root_train.to_numpy()
    root_test = root_test.to_numpy()

    data = np.zeros((root_train.shape[0]+root_test.shape[0],root_train.shape[1]))
    lentgh_train = len(root_train)
    lentgh_test = len(root_test)
    for idx in range(0, lentgh_train):
        data[idx] = root_train[idx]
    for idx in range(0, lentgh_test):
        data[idx+lentgh_train] = root_test[idx]
    data = pd.DataFrame(data, columns=columns)

    sgn = data[data["class"] == 1]
    bkg = data[data["class"] == 0]

    # remove previous file
    try:
        os.remove(f"/cephfs/user/s6herose/Bachelorarbeit/Programmieren/codemasterthesis/finalScripts/distance/plots/{save_directory}/signal_{conversion_type}_NN_results.root")
    except OSError:
        pass

    try:
        os.remove(f"/cephfs/user/s6herose/Bachelorarbeit/Programmieren/codemasterthesis/finalScripts/distance/plots/{save_directory}/background_{conversion_type}_NN_results.root")
    except OSError:
        pass
 
    # create root file for signal and background
    sgn_root = up.recreate(f"/cephfs/user/s6flkirf/master_thesis/codemasterthesis/finalScripts/distance/plots/{save_directory}/signal_{conversion_type}_NN_results.root", compression=up.ZLIB(4))
    sgn_root["SinglePhoton"] = up.newtree({ "class": np.float64 , "y_pt": np.float64 , "y_eta": np.float64 , "y_isTruthMatchedPhoton": np.float64 , "y_ptcone20": np.float64 ,
                                        "y_ptcone40": np.float64 , "y_topoetcone20": np.float64 , "y_topoetcone40": np.float64 , "y_IsLoose": np.float64 , "y_IsTight": np.float64 , "y_convType": np.float64 ,
                                        "dataWeightPtBin": np.float64 , "evt_mu": np.float64 , "y_pred": np.float64 , "mcTotWeight": np.float64})
    sgn_root["SinglePhoton"].extend({ "class": sgn["class"] , "y_pt": sgn["y_pt"] , "y_eta": sgn["y_eta"] , "y_isTruthMatchedPhoton": sgn["y_isTruthMatchedPhoton"] , "y_ptcone20": sgn["y_ptcone20"] ,
                                        "y_ptcone40": sgn["y_ptcone40"] , "y_topoetcone20": sgn["y_topoetcone20"] , "y_topoetcone40": sgn["y_topoetcone40"] , "y_IsLoose": sgn["y_IsLoose"] , "y_IsTight": sgn["y_IsTight"] , "y_convType": sgn["y_convType"] ,
                                        "dataWeightPtBin": sgn["dataWeightPtBin"] , "evt_mu": sgn["evt_mu"] , "y_pred": sgn["y_pred"] , "mcTotWeight": sgn["mcTotWeight"]})

    bkg_root = up.recreate(f"/cephfs/user/s6flkirf/master_thesis/codemasterthesis/finalScripts/distance/plots/{save_directory}/background_{conversion_type}_NN_results.root", compression=up.ZLIB(4))
    bkg_root["SinglePhoton"] = up.newtree({ "class": np.float64 , "y_pt": np.float64 , "y_eta": np.float64 , "y_isTruthMatchedPhoton": np.float64 , "y_ptcone20": np.float64 ,
                                        "y_ptcone40": np.float64 , "y_topoetcone20": np.float64 , "y_topoetcone40": np.float64 , "y_IsLoose": np.float64 , "y_IsTight": np.float64 , "y_convType": np.float64 ,
                                        "dataWeightPtBin": np.float64 , "evt_mu": np.float64 , "y_pred": np.float64 , "mcTotWeight": np.float64})
    bkg_root["SinglePhoton"].extend({ "class": bkg["class"] , "y_pt": bkg["y_pt"] , "y_eta": bkg["y_eta"] , "y_isTruthMatchedPhoton": bkg["y_isTruthMatchedPhoton"] , "y_ptcone20": bkg["y_ptcone20"] ,
                                        "y_ptcone40": bkg["y_ptcone40"] , "y_topoetcone20": bkg["y_topoetcone20"] , "y_topoetcone40": bkg["y_topoetcone40"] , "y_IsLoose": bkg["y_IsLoose"] , "y_IsTight": bkg["y_IsTight"] , "y_convType": bkg["y_convType"] ,
                                        "dataWeightPtBin": bkg["dataWeightPtBin"] , "evt_mu": bkg["evt_mu"] , "y_pred": bkg["y_pred"] , "mcTotWeight": bkg["mcTotWeight"]})

