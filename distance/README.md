## Binary classifier neural network

This neural network can be used for the distance correlation classifier for the photon ID. This classifier combines the Loss functions in the following way: 

L_combined = L_Classifier + lambda L_distanceCorrelation

In order to run it use the following command:
 
```
python runDistance.py config.ini
```

All settings can be specified in the config file. Available options are specified below. The config file is divided into different blocks. Stick to this in order to run everything correctly.

1. [general]
- save_directory : string for the save directory
- seed : integer to set random seed for numpy 
- verbose : verbose level for tensorflow model.fit output (options: [0,1,2])
- reweighting : leave pt and eta as they are or reweight background to signal (options: [normal, reweight]) 
- entry_stop: integer to decide how many events will be used for signal and background
- training_variables : list of all input variables that shall be used for training
- conversion_type : conversion type of the data (options: [converted, unconverted])
- selection_type : apply additional selection (options: [normal, pt_cut, eta_cut, both_cut])
- selection_list : only used if selection_type != normal specify 4 floats that will be used in the following way: [lower_pt_cut, upper_pt_cut, lower_eta_cut, upper_eta_cut]
- scaler_type : Scaler to transform the input variables (options: [MinMax, Standard])
- split_size : float so select the split percentage between training and test
- save_scaler : Save scaler into the save directory (options: [False, True])
- save_model : Save model into the save directory (options: [False, True])
- plot_type : = 1 will produce plots that fit perfectly into one line / = 2 will produce quadratic plots to align two plots horizontically
- training_mode : Run automatic optimization or normal model (options: [normal, optimize])

2. [classifier]
- classifier_epochs : integer to specify the number of epochs
- classifier_batch_size : integer to specify the batch size
- classifier_layers : list with nodes per layer
- classifier_dropout : list with dropout percentage per layer (needs to have the same size as classifier_layers)
- classifier_learning_rate : float to specify the learning rate
- classifier_optimizer_type : optimizer for training (options: [Adam, SGD])
- classifier_use_dropout : decide whether to use dropout layers (options: [False, True])
- classifier_use_batch_norm : decide whether to use batch normalizations layers (options: [False, True])
- classifier_early_stopping : decide whether to use early stopping (options: [False, True])
- classifier_pre_training : decide whether to run pre-training on the classifier (options: [False, True])
- classifier_pre_epochs = integer to specify the pretraining epochs

3. [decorrelation]
- use_zeros = decide whether to use events that have y_ptcone40=0 for the deorrelation (use_zeros=True) or not (use_zeros=False) in the second case the weights of all y_ptcone40=0 events will be set to 0
- classifier_lambda_value = float to specify the lambda value (L_C+lambda L_DistanceCorrelation) for the classifier

4. [optimize]
- num_steps : integer to define training steps in bayesian optimization
- num_nodes : list with three elements so specify the nodes per layer search space in the following form: [smalles_num,largest_num,step_size]
- num_layers : list with three elements so specify the number of layer search space in the following form: [smalles_num,largest_num,step_size]
- dropout : list with three elements so specify the dropout percentage per layer search space in the following form: [smalles_num,largest_num,step_size]
- leraning_rate : list to specify the learning rate search space
- lambda_value : list to specify the lambda value (L_C+lambda L_DistanceCorrelation) search space
5. [save] 
- save_to_root : save needed results to a root file in the save_directory (options: [False, True])
- save_to_numpy : save needed results to numpy file in the save_directory (options: [False, True])

An example config is given below:
```
# general setting
[general]
save_directory = test_optimize
seed = 42
verbose = 1
reweighting = normal
entry_stop = 4000
training_variables = y_Reta,y_Rphi,y_weta1,y_weta2,y_deltae,y_fracs1,y_Eratio,y_wtots1,y_Rhad,y_Rhad1,y_f1,y_e277,y_pt,y_eta
conversion_type = converted
selection_type = normal
selection_list = 0.0,100,0.0,0.6
scaler_type = Standard
split_size = 0.4
save_scaler = True
save_model =True
plot_type = 1
training_mode = optimize

# classifier settings
[classifier]
classifier_epochs = 100
classifier_batch_size = 16384
classifier_layers = 100,80,80
classifier_dropout = 0.3,0.4,0.4
classifier_learning_rate = 0.001
classifier_optimizer_type = Adam
classifier_use_dropout = False
classifier_use_batch_norm = True
classifier_early_stopping = True
classifier_pre_training = False
classifier_pre_epochs = 10

# decorrelation settings
[decorrelation]
use_zeros = True
classifier_lambda_value = 3

# hyperparameter optimizer settings
[optimize]
num_steps = 20
num_nodes = 32,128,32
num_layers = 1,6,1
dropout = 0.1,0.5,0.1
leraning_rate = 0.01,0.001,0.0001
lambda_value = 10

# save results
[save]
save_to_root = True
save_to_numpy = False
```
