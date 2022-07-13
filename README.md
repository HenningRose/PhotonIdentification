## Prediction on data

This script can be used to predict the class (signal/background) for data. The resulting root file will be saved in the subdirectory of the model that is used for predictiing.

Example: we use a classifier model named test to predict on data15 -> The data predictions will be saved in the /classifier/plots/test/ directory as data15_conversion_type_NN_results.root.

To apply the scaler to the correct variables (shower shape + pt and eta or just shower shape) it will load the config file out of the /classifier/plots/test/ directory. **SO MAKE SURE THAT THIS CONFIG IS NAMED "config.ini" AND CONTAINS THE "training_variables" ENTRY IN THE "general" block!**

In order to run it use the following command:
 
```
python runPrediction.py config.ini
```

All settings can be specified in the config file. Available options are specified below. The config file is divided into different blocks. Stick to this in order to run everything correctly.

1. [prediction]
- data_file : list of data files to predict on (options: [data15, data16, data17, data18])
- folder : model to load in
- network_type : network types that decides where to load the models from (options: [classifier, distance, ann]) The last two are not implemented yet!
- conversion_type : conversion type of the data (options: [converted, unconverted])

An example config is given below:
```
[prediction]
data_files = data15,data16
folder = test_save
network_type = classifier
conversion_type = converted
```
