import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import chisquare
from scipy.spatial import distance
from scipy import integrate
from sklearn.metrics import roc_curve, auc, r2_score
import sys

""" 
This script contains all functions that are used to create all pre training plots in the distance correlation network.
"""
plt.rcParams["font.family"] = "serif"
fig_width_pt = 437.46118  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inches
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height =fig_width*golden_mean       # height in inches
rectangular_fig_size = [fig_width,fig_height]
quadratic_fig_size = [fig_width,fig_width]



###########################################
####### CLASSIFIER RESULTS PLOTS ##########
###########################################

def Loss(conversion, model_history, save_directory):
    """ Plot the classifier loss
    Arguments:
    conversion: conversion type
    model_history; output history from model.fit(...)
    save_directory: directory to save plot in
    """
    train_train = model_history.history['loss']
    train_test = model_history.history['val_loss']

    label_size = 10
    fig_size = rectangular_fig_size
    plt.rcParams['legend.title_fontsize'] = label_size
    fig, axs = plt.subplots(1, 1, figsize=fig_size)

    axs.tick_params(which='major', length=10)
    axs.tick_params(which='minor', length=5)
    axs.yaxis.set_minor_locator(AutoMinorLocator(4))
    axs.xaxis.set_minor_locator(AutoMinorLocator(4))
    axs.tick_params(axis='both', which='major', labelsize=10)
    axs.tick_params(axis='x', direction='in', bottom=True, labelbottom=True, top=True, labeltop=False, which='both')
    axs.tick_params(axis='y', direction='in', left=True, labelleft=True, right=True, labelright=False, which='both')
    axs.plot(train_train, color='cornflowerblue', lw=1.5, label='training')
    axs.plot(train_test, color='darkorange', lw=1.5, label='validation')
    axs.set_ylabel(r"$L_{classifier}$", fontsize=label_size)
    axs.set_xlabel('epochs', fontsize=label_size)
    axs.grid(True)
    #axs.set_title('classifier loss', fontsize=label_size)
    legend = axs.legend(title=f'{conversion} photons', fontsize=label_size, loc="best", facecolor='grey', edgecolor='black', framealpha=0.1)
    legend._legend_box.sep = 5

    #fig.tight_layout()
    plt.savefig(f"/cephfs/user/s6herose/Bachelorarbeit/Programmieren/codemasterthesis/finalScripts/distance/plots/{save_directory}/pre/pre_loss.pdf", bbox_inches='tight')
    plt.close


def ROC(conversion, y_train, y_train_pred, w_train, y_test, y_test_pred, w_test, plot_type, save_directory):
    """ Plot the classifier ROC curve
    Arguments:
    conversion: conversion type
    y_train, y_test: true targets for test and training sample
    y_train_pred, y_test_pred: predicted targets for test and training sample
    w_train, w_test: weights for training and test sample
    plot_type: plot 1 or 2 plots per line in latex file
    save_directory: directory to save plot in
    """

    fpr_train, tpr_train, threshold_train = roc_curve(y_train, y_train_pred, sample_weight=w_train)
    fpr_test, tpr_test, threshold_test = roc_curve(y_test, y_test_pred, sample_weight=w_test)
    
    # reordering in case of negative weights
    a=np.column_stack([fpr_train, tpr_train])
    ind=np.argsort(a[:,0])
    fpr_train=a[ind][:,0]
    tpr_train=a[ind][:,1]

    a=np.column_stack([fpr_test, tpr_test])
    ind=np.argsort(a[:,0])
    fpr_test=a[ind][:,0]
    tpr_test=a[ind][:,1]

    auc_train = auc(fpr_train, tpr_train)
    auc_test = auc(fpr_test, tpr_test)

    if (plot_type == 1):
        label_size = 10
        fig_size = rectangular_fig_size
        plt.rcParams['legend.title_fontsize'] = label_size
    elif (plot_type == 2):
        label_size = 12
        fig_size = quadratic_fig_size
        plt.rcParams['legend.title_fontsize'] = label_size
    else:
        sys.exit("This plot_type is not available! Use 1 for rectangluar or 2 for quadratic plots")
    
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=fig_size)

    ax1.plot(fpr_train, tpr_train, color='cornflowerblue', lw=1.5, label='training ROC curve (AUC = {:.3f})'.format(auc_train))
    ax1.plot(fpr_test, tpr_test, color='darkorange', lw=1.5, label='test ROC curve (AUC = {:.3f})'.format(auc_test))
    ax1.plot([0, 1], [0, 1], 'k--', lw=1.5)
    ax1.set_xlabel('False Positive Rate (FPR)', fontsize=label_size)
    ax1.set_ylabel('True Positive Rate (TPR)', fontsize=label_size)
    ax1.set_xlim(0, 1.02)
    ax1.set_ylim(0, 1.02)
    ax1.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.tick_params(which='major', length=10)
    ax1.tick_params(which='minor', length=5)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax1.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax1.tick_params(axis='x', direction='in', bottom=True, labelbottom=True, top=True, labeltop=False, which='both')
    ax1.tick_params(axis='y', direction='in', left=True, labelleft=True, right=True, labelright=False, which='both')
    legend = ax1.legend(title=f'{conversion} photons', fontsize=label_size, loc="best", facecolor='grey', edgecolor='black', framealpha=0.1)
    legend._legend_box.sep = 5
    ax1.grid(True)
    #ax1.set_title('classifier ROC curve', fontsize=label_size)
    #fig.tight_layout()
    plt.savefig(f"/cephfs/user/s6herose/Bachelorarbeit/Programmieren/codemasterthesis/finalScripts/distance/plots/{save_directory}/pre/pre_ROC.pdf", bbox_inches='tight')
    plt.close


def classifierPredictions(conversion, y_train, y_train_pred, w_train, y_test, y_test_pred, w_test, plot_type, save_directory):
    """ 
    plots the classifier response for training and test
    Arguments:
    conversion: conversion type
    y_train, y_test: true targets for test and training sample
    y_train_pred, y_test_pred: predicted targets for test and training sample
    w_train, w_test: weights for test and training sample
    plot_type: plot 1 or 2 plots per line in latex file
    save_directory: directory to save plot in
    """
    if (plot_type == 1):
        label_size = 10
        fig_size = rectangular_fig_size
        plt.rcParams['legend.title_fontsize'] = label_size
    elif (plot_type == 2):
        label_size = 12
        fig_size = quadratic_fig_size
        plt.rcParams['legend.title_fontsize'] = label_size
    else:
        sys.exit("This plot_type is not available! Use 1 for rectangluar or 2 for quadratic plots")

    fig_train, ax1_train = plt.subplots(nrows=1, ncols=1, figsize=fig_size)
    fig_test, ax1_test = plt.subplots(nrows=1, ncols=1, figsize=fig_size)
        
    # divide data in signal and background
    train_frame = pd.DataFrame()
    test_frame = pd.DataFrame()

    train_frame['y_true'] = y_train
    train_frame['y_pred'] = y_train_pred
    train_frame['weights']= w_train
    test_frame['y_true'] = y_test
    test_frame['y_pred'] = y_test_pred
    test_frame['weights']= w_test

    train_sgn = train_frame[train_frame['y_true']==1].y_pred.values
    train_bkg = train_frame[train_frame['y_true']==0].y_pred.values
    w_train_sgn = train_frame[train_frame['y_true']==1].weights.values
    w_train_bkg = train_frame[train_frame['y_true']==0].weights.values
    test_sgn = test_frame[test_frame['y_true']==1].y_pred.values
    test_bkg = test_frame[test_frame['y_true']==0].y_pred.values
    w_test_sgn = test_frame[test_frame['y_true']==1].weights.values
    w_test_bkg = test_frame[test_frame['y_true']==0].weights.values

    # plot distributions
    #fig, ax1 = plt.subplots(nrows=1, ncols=2, figsize=(11.69,8.27))

    ax1_train.hist(train_sgn, weights=w_train_sgn, bins=50, color="darkorange", edgecolor='darkorange', linewidth=1.5, histtype="step", label="signal", range=(0.,1.), density=True, stacked=True)
    ax1_train.hist(train_bkg, weights=w_train_bkg, bins=50, color="cornflowerblue", edgecolor='cornflowerblue', linewidth=1.5, histtype="step", label="background", range=(0.,1.), density=True, stacked=True) 
    ax1_train.set_xlabel('classifier response', fontsize=label_size)
    ax1_train.set_ylabel('weighted events', fontsize=label_size)
    ax1_train.set_xlim(0, 1)
    ax1_train.tick_params(which='major', length=10)
    ax1_train.tick_params(which='minor', length=5)
    ax1_train.tick_params(axis='both', which='major', labelsize=10)
    ax1_train.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1_train.set_xticklabels(ax1_train.get_xticks(), rotation = 45)
    ax1_train.set_yticks([])
    ax1_train.set_yticks([], minor=True)
    ax1_train.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax1_train.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax1_train.tick_params(axis='x', direction='in', bottom=True, labelbottom=True, top=True, labeltop=False, which='both')
    ax1_train.tick_params(axis='y', direction='in', left=True, labelleft=True, right=True, labelright=False, which='both')
    legend = ax1_train.legend(title=f'{conversion} photons - training data', loc="best", facecolor='grey', edgecolor='black', framealpha=0.1)
    legend._legend_box.sep = 5
    #ax1_train.set_title('training data', fontsize=label_size)

    ax1_test.hist(test_sgn, weights=w_test_sgn, bins=50, color="darkorange", edgecolor='darkorange', linewidth=1.5, histtype="step", label="signal", range=(0.,1.), density=True, stacked=True)
    ax1_test.hist(test_bkg, weights=w_test_bkg, bins=50, color="cornflowerblue", edgecolor='cornflowerblue', linewidth=1.5, histtype="step", label="background", range=(0.,1.), density=True, stacked=True) 
    ax1_test.set_xlabel('classifier response', fontsize=label_size)
    ax1_test.set_ylabel('weighted events', fontsize=label_size)
    ax1_test.set_xlim(0, 1)
    ax1_test.tick_params(which='major', length=10)
    ax1_test.tick_params(which='minor', length=5)
    ax1_test.tick_params(axis='both', which='major', labelsize=10)
    ax1_test.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1_test.set_xticklabels(ax1_test.get_xticks(), rotation = 45)
    ax1_test.set_yticks([])
    ax1_test.set_yticks([], minor=True)
    ax1_test.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax1_test.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax1_test.tick_params(axis='x', direction='in', bottom=True, labelbottom=True, top=True, labeltop=False, which='both')
    ax1_test.tick_params(axis='y', direction='in', left=True, labelleft=True, right=True, labelright=False, which='both')
    legend = ax1_test.legend(title=f'{conversion} photons - test data', loc="best", facecolor='grey', edgecolor='black', framealpha=0.1)
    legend._legend_box.sep = 5
    #ax1_test.set_title('test data', fontsize=label_size)
    #fig.tight_layout()
    fig_train.savefig(f"/cephfs/user/s6herose/Bachelorarbeit/Programmieren/codemasterthesis/finalScripts/distance/plots/{save_directory}/pre/pre_score_train.pdf", bbox_inches='tight')
    plt.close(fig_train)
    fig_test.savefig(f"/cephfs/user/s6herose/Bachelorarbeit/Programmieren/codemasterthesis/finalScripts/distance/plots/{save_directory}/pre/pre_score_test.pdf", bbox_inches='tight')
    plt.close(fig_test)
        
###########################################
#### CORRELATION AND EFFICIENCY PLOTS #####
###########################################

def Efficiency(conversion, y_train, y_train_pred, w_train, tight_train, y_test, y_test_pred, w_test, tight_test, save_directory):
    """ 
    plots the classifier efficieny for training and test and compare it with tight ID WP
    Arguments:
    conversion: conversion type
    y_train, y_test: true targets for test and training sample
    y_train_pred, y_test_pred: predicted targets for test and training sample
    w_train, w_test: weights for test and training sample
    tight_train, tight_test: tight ID for test and training sample
    save_directory: directory to save plot in
    """
    train_frame = pd.DataFrame()
    test_frame = pd.DataFrame()

    train_frame['y_true'] = y_train
    train_frame['y_pred'] = y_train_pred
    train_frame['mcTotWeight']= w_train
    train_frame['y_IsTight']= tight_train
    test_frame['y_true'] = y_test
    test_frame['y_pred'] = y_test_pred
    test_frame['mcTotWeight']= w_test
    test_frame['y_IsTight']= tight_test

    bkg_eff_train, sgn_eff_train, threshold_train = roc_curve(train_frame['y_true'], train_frame['y_pred'], sample_weight=train_frame['mcTotWeight'])
    bkg_eff_test, sgn_eff_test, threshold_test = roc_curve(test_frame['y_true'], test_frame['y_pred'], sample_weight=test_frame['mcTotWeight'])
    
    # reordering in case of negative weights
    a=np.column_stack([bkg_eff_train, sgn_eff_train])
    ind=np.argsort(a[:,0])
    bkg_eff_train=a[ind][:,0]
    sgn_eff_train=a[ind][:,1]

    a=np.column_stack([bkg_eff_test, sgn_eff_test])
    ind=np.argsort(a[:,0])
    bkg_eff_test=a[ind][:,0]
    sgn_eff_test=a[ind][:,1]

    auc_train = auc(bkg_eff_train, sgn_eff_train)
    auc_test = auc(bkg_eff_test, sgn_eff_test)

    bkg_rej_train = 1-bkg_eff_train
    bkg_rej_test = 1-bkg_eff_test

    def tightWorkingPoint(data):
        sgn_pass_tight = data[(data['y_true']==1) & (data['y_IsTight']==1)]['mcTotWeight']
        bkg_pass_tight = data[(data['y_true']==0) & (data['y_IsTight']==1)]['mcTotWeight']
        sgn_pass_tight_sum = sgn_pass_tight.sum()
        bkg_pass_tight_sum = bkg_pass_tight.sum()
        sgn = data[data['y_true']==1]['mcTotWeight']
        bkg = data[data['y_true']==0]['mcTotWeight']
        sgn_sum = sgn.sum()
        bkg_sum = bkg.sum()

        sgn_eff_WP = sgn_pass_tight_sum/sgn_sum
        bkg_rej_WP = 1 - (bkg_pass_tight_sum/bkg_sum)

        return sgn_eff_WP, bkg_rej_WP
  
    x_WP_train, y_WP_train = tightWorkingPoint(train_frame)
    x_WP_test, y_WP_test = tightWorkingPoint(test_frame)

    label_size = 10
    fig_size = rectangular_fig_size
    plt.rcParams['legend.title_fontsize'] = label_size
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=fig_size)

    # plot test and train efficiencys
    plt.plot(sgn_eff_train, bkg_rej_train, label='training data (AUC = {:.3f})'.format(auc_train), lw=1.5, color='cornflowerblue')
    plt.plot(sgn_eff_test, bkg_rej_test, label='test data (AUC = {:.3f})'.format(auc_test), lw=1.5, color='darkorange')
    

    # plot tight working point
    ax1.vlines(x_WP_train, 0, y_WP_train, linestyle="dotted", color='black', lw=1.5,)
    ax1.hlines(y_WP_train, 0, x_WP_train, linestyle="dotted", color='black', lw=1.5,)
    ax1.plot(x_WP_train, y_WP_train, '.', label='tight WP', color='black', markersize=4)
    #ax1.plot(x_WP_test, y_WP_test, '.', label='tight WP_test', color='black', markersize=4)

    ax1.set_ylabel(r'background rejection $b_{R}$', fontsize=label_size)
    ax1.set_xlabel(r'signal efficiency $s_{E}$', fontsize=label_size)
    ax1.set_xlim(0, 1.02)
    ax1.set_ylim(0, 1.02)
    ax1.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_xticklabels(ax1.get_xticks(), rotation = 45)
    ax1.tick_params(which='major', length=10)
    ax1.tick_params(which='minor', length=5)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax1.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax1.tick_params(axis='x', direction='in', bottom=True, labelbottom=True, top=True, labeltop=False, which='both')
    ax1.tick_params(axis='y', direction='in', left=True, labelleft=True, right=True, labelright=False, which='both')
    legend = ax1.legend(title=f'{conversion} photons', loc="best", facecolor='grey', edgecolor='black', framealpha=0.1)
    legend._legend_box.sep = 5
    ax1.grid(True)
    #ax1.set_title('classifier ROC curve', fontsize=16)
    
    fig.tight_layout()
    plt.savefig(f"/cephfs/user/s6herose/Bachelorarbeit/Programmieren/codemasterthesis/finalScripts/distance/plots/{save_directory}/pre/pre_efficiency.pdf", bbox_inches='tight')
    plt.close()

def JSD(conversion, y_train, y_train_pred, w_train, tight_train, iso_train, y_test, y_test_pred, w_test, tight_test, iso_test, plot_type, save_directory):
    """ 
    plots the classifier 1/JSD for background events failing and passing different cuts for training and test set and compares it with tight ID WP
    Arguments:
    conversion_ conversion type
    y_train, y_test: true targets for test and training sample
    y_train_pred, y_test_pred: predicted targets for test and training sample
    w_train, w_test: weights for test and training sample
    tight_train, tight_test: tight ID for test and training sample
    iso_train, iso_test: isolation variable for test and training sample
    plot_type: plot 1 or 2 plots per line in latex file
    save_directory: directory to save plot in
    """
    if (plot_type == 1):
        label_size = 10
        fig_size = rectangular_fig_size
        plt.rcParams['legend.title_fontsize'] = label_size
    elif (plot_type == 2):
        label_size = 12
        fig_size = quadratic_fig_size
        plt.rcParams['legend.title_fontsize'] = label_size
    else:
        sys.exit("This plot_type is not available! Use 1 for rectangluar or 2 for quadratic plots")
    

    train_frame = pd.DataFrame()
    test_frame = pd.DataFrame()

    train_frame['y_true'] = y_train
    train_frame['y_pred'] = y_train_pred
    train_frame['mcTotWeight']= w_train
    train_frame['y_IsTight']= tight_train
    train_frame['track']= iso_train
    test_frame['y_true'] = y_test
    test_frame['y_pred'] = y_test_pred
    test_frame['mcTotWeight']= w_test
    test_frame['y_IsTight']= tight_test
    test_frame['track']= iso_test

    train_bkg = train_frame[train_frame['y_true']==0]
    test_bkg = test_frame[test_frame['y_true']==0]
        
    def backgroundRejection(cut, bkg_pred):
        bkg = bkg_pred['mcTotWeight'].sum()
        bkg_cut_arr = bkg_pred[bkg_pred['y_pred'] > cut]
        bkg_cut = bkg_cut_arr['mcTotWeight'].sum()
        bkg_eff = bkg_cut/bkg
        bkg_rej = 1 - bkg_eff
        return bkg_rej

    def tightRejection(data):
        bkg_pass_tight = data[(data['y_true']==0) & (data['y_IsTight']==1)]['mcTotWeight']
        bkg_pass_tight_sum = bkg_pass_tight.sum()
        bkg = data[data['y_true']==0]['mcTotWeight']
        bkg_sum = bkg.sum()
        bkg_rej_WP = 1 - (bkg_pass_tight_sum/bkg_sum)

        return bkg_rej_WP

    def getBinHeight(data, weights, n_bins, x_min, x_max):
        heights, edges = np.histogram(data, bins=n_bins, range=(x_min, x_max), weights=weights)
        norm_heights = heights/heights.sum()
        return norm_heights

    array = np.linspace(0.01,0.99,20)
    # calculate rejctions and WP
    y_test = [backgroundRejection(i, test_bkg) for i in array]
    y_train = [backgroundRejection(i, train_bkg) for i in array]
    bkg_rej_WP = tightRejection(train_bkg)

    

    
    train_JSD_pass_inclusive = []
    test_JSD_pass_inclusive = []
    train_JSD_fail_inclusive = []
    test_JSD_fail_inclusive = []
    cut_values = []

    tight_pass = train_bkg[train_bkg['y_IsTight'] == 1]
    tight_fail = train_bkg[train_bkg['y_IsTight'] == 0]
    tight_heights_pass = getBinHeight(tight_pass['track'], tight_pass['mcTotWeight'], 60, 0, 150)
    tight_heights_fail = getBinHeight(tight_fail['track'], tight_fail['mcTotWeight'], 60, 0, 150)

    y_train_heights_inclusive = getBinHeight(train_bkg['track'], train_bkg['mcTotWeight'], 60, 0, 150)
    y_test_heights_inclusive = getBinHeight(test_bkg['track'], test_bkg['mcTotWeight'], 60, 0, 150)

    # JSD background only for tight ID 
    tight_JSD_fail_inclusive = distance.jensenshannon(y_train_heights_inclusive, tight_heights_fail, 2.0)
    tight_JSD_pass_inclusive = distance.jensenshannon(y_train_heights_inclusive, tight_heights_pass, 2.0)

    # calculate JSD for background only
    for cut in array:
       
        train_pass = train_bkg[train_bkg['y_pred'] >= cut]
        train_fail = train_bkg[train_bkg['y_pred'] < cut]
        test_pass = test_bkg[test_bkg['y_pred'] >= cut]
        test_fail = test_bkg[test_bkg['y_pred'] < cut]
        
        y_train_heights_pass = getBinHeight(train_pass['track'], train_pass['mcTotWeight'], 60, 0, 150)
        y_train_heights_fail = getBinHeight(train_fail['track'], train_fail['mcTotWeight'], 60, 0, 150)
        y_test_heights_pass = getBinHeight(test_pass['track'], test_pass['mcTotWeight'], 60, 0, 150)
        y_test_heights_fail = getBinHeight(test_fail['track'], test_fail['mcTotWeight'], 60, 0, 150)        
        
        cut_values.append(cut)
        # JSD classifier
        train_JSD_fail_inclusive.append(distance.jensenshannon(y_train_heights_inclusive, y_train_heights_fail, 2.0))
        test_JSD_fail_inclusive.append(distance.jensenshannon(y_test_heights_inclusive, y_test_heights_fail, 2.0))
        train_JSD_pass_inclusive.append(distance.jensenshannon(y_train_heights_inclusive, y_train_heights_pass, 2.0))
        test_JSD_pass_inclusive.append(distance.jensenshannon(y_test_heights_inclusive, y_test_heights_pass, 2.0))


    # 1/JSD classifier
    train_inv_JSD_pass_inclusive = [1/JSD for JSD in train_JSD_pass_inclusive]
    test_inv_JSD_pass_inclusive = [1/JSD for JSD in test_JSD_pass_inclusive]
    train_inv_JSD_fail_inclusive = [1/JSD for JSD in train_JSD_fail_inclusive]
    test_inv_JSD_fail_inclusive = [1/JSD for JSD in test_JSD_fail_inclusive]

    # 1/JSD tight ID 
    tight_inv_JSD_fail_inclusive = 1/tight_JSD_fail_inclusive 
    tight_inv_JSD_pass_inclusive = 1/tight_JSD_pass_inclusive


    fig_fail, ax1_fail = plt.subplots(nrows=1, ncols=1, figsize=fig_size)
    fig_pass, ax1_pass = plt.subplots(nrows=1, ncols=1, figsize=fig_size)

    ax1_pass.plot(y_train, train_inv_JSD_pass_inclusive, color='cornflowerblue', marker='o', linestyle='dashed', linewidth=1.5, markersize=4, label='training data')
    ax1_pass.plot(y_test, test_inv_JSD_pass_inclusive, color='darkorange', marker='o', linestyle='dashed', linewidth=1.5, markersize=4, label='test data')
    ax1_pass.plot(bkg_rej_WP, tight_inv_JSD_pass_inclusive, color='black', marker='o', linestyle='dashed', linewidth=1.5, markersize=4, label='tight ID')
    ax1_pass.set_xlabel(r'background rejection $b_{R}$', fontsize=label_size)
    ax1_pass.set_ylabel(r'$1/JSD_{pass\:vs\:inclusive}$', fontsize=label_size)
    ax1_pass.set_xlim(0, 1)
    ax1_pass.set_ylim(1, 1000)
    ax1_pass.set_yscale('log')
    ax1_pass.tick_params(which='major', length=10)
    ax1_pass.tick_params(which='minor', length=5)
    ax1_pass.tick_params(axis='both', which='major', labelsize=10)
    #ax1_pass.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax1_pass.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax1_pass.tick_params(axis='x', direction='in', bottom=True, labelbottom=True, top=True, labeltop=False, which='both')
    ax1_pass.tick_params(axis='y', direction='in', left=True, labelleft=True, right=True, labelright=False, which='both')
    ax1_pass.grid(b=None, which='both', axis='y', linestyle='dashed', alpha=0.5)
    ax1_pass.grid(b=None, which='major', axis='x', linestyle='dashed', alpha=0.5)
    legend = ax1_pass.legend(title=f'{conversion} photons', loc="best", facecolor='grey', edgecolor='black', framealpha=0.1)
    legend._legend_box.sep = 5
    #ax1_pass.set_title('training data - {}'.format(title), fontsize=10)

    ax1_fail.plot(y_train, train_inv_JSD_fail_inclusive, color='cornflowerblue', marker='o', linestyle='dashed', linewidth=1.5, markersize=4, label='training data')
    ax1_fail.plot(y_test, test_inv_JSD_fail_inclusive, color='darkorange', marker='o', linestyle='dashed', linewidth=1.5, markersize=4, label='test data')
    ax1_fail.plot(bkg_rej_WP, tight_inv_JSD_fail_inclusive, color='black', marker='o', linestyle='dashed', linewidth=1.5, markersize=4, label='tight ID')
    ax1_fail.set_xlabel(r'background rejection $b_{R}$', fontsize=label_size)
    ax1_fail.set_ylabel(r'$1/JSD_{fail\:vs\:inclusive}$', fontsize=label_size)
    ax1_fail.set_xlim(0, 1)
    ax1_fail.set_ylim(1, 1000)
    ax1_fail.set_yscale('log')
    ax1_fail.tick_params(which='major', length=10)
    ax1_fail.tick_params(which='minor', length=5)
    ax1_fail.tick_params(axis='both', which='major', labelsize=10)
    #ax1_fail.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax1_fail.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax1_fail.tick_params(axis='x', direction='in', bottom=True, labelbottom=True, top=True, labeltop=False, which='both')
    ax1_fail.tick_params(axis='y', direction='in', left=True, labelleft=True, right=True, labelright=False, which='both')
    ax1_fail.grid(b=None, which='both', axis='y', linestyle='dashed', alpha=0.5)
    ax1_fail.grid(b=None, which='major', axis='x', linestyle='dashed', alpha=0.5)
    legend = ax1_fail.legend(title=f'{conversion} photons', loc="best", facecolor='grey', edgecolor='black', framealpha=0.1)
    legend._legend_box.sep = 5
    #ax1_fail.set_title('training data - {}'.format(title), fontsize=label_size)

    fig_fail.savefig(f"/cephfs/user/s6herose/Bachelorarbeit/Programmieren/codemasterthesis/finalScripts/distance/plots/{save_directory}/pre/pre_JSD_fail.pdf", bbox_inches='tight')
    plt.close(fig_fail)
    fig_pass.savefig(f"/cephfs/user/s6herose/Bachelorarbeit/Programmieren/codemasterthesis/finalScripts/distance/plots/{save_directory}/pre/pre_JSD_pass.pdf", bbox_inches='tight')
    plt.close(fig_pass)