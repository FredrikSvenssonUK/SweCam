"""
LGBM predictor for the Tox24 challenge.
https://github.com/FredrikSvenssonUK/SweCam/blob/main/LICENSE

LightGBM: An Effective and Scalable Algorithm for Prediction of Chemical Toxicity–Application to the Tox21 and Mutagenicity Data Sets
Jin Zhang, Daniel Mucs, Ulf Norinder, and Fredrik Svensson
Journal of Chemical Information and Modeling 2019 59 (10), 4150-4158
DOI: 10.1021/acs.jcim.9b00633

https://github.com/microsoft/LightGBM
"""


__author__ = "Ulf Norinder, Fredrik Svensson"
__date__ = "15/08/24"


### Imports ###
from optparse import OptionParser

import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np


### Config ###
#Process commandline options:
usage = "usage: %prog -i inputfile [optional args]"
parser = OptionParser(usage=usage)
parser.add_option("-i","--in", dest="infile",help="Training set file. (required)", metavar ="file")
parser.add_option("-t","--test", dest="test",help="Test set file. (optional)", metavar ="file")
parser.add_option("-d","--delim", dest="delim",help="Delimiter in input",default='\t')
parser.add_option("-f","--header", dest="header",help="Set to row of header in the infile or None.",default=None)
(options,args) = parser.parse_args()

# set configs
outfile = "lgbm_predictions.csv" # save predictions to this file, only used for external set
random_seed = 42
folds = 10 # number of cross validation folds


### Functions ###

def lgbm_model():
    
    param_dict = {'learning_rate': 0.005117712512654444, 'n_estimators': 900, 'lambda_l1': 6.152805124789538e-08, 'lambda_l2': 6.580329064493613e-05, 'num_leaves': 142, 'feature_fraction': 0.9338378651874101, 'bagging_fraction': 0.5617248336617564, 'bagging_freq': 5, 'min_child_samples': 10}

    model = lgb.LGBMRegressor(objective = 'regression',
                              verbosity=-1,
                              boosting_type = "gbdt",
                              random_state=random_seed,
                              **param_dict,
                              )
    
    return model


### Main ###

if __name__ == "__main__":

    # Load data
    data = pd.read_csv(options.infile, sep=options.delim, header=options.header)
    X = data.iloc[:,2:]
    X = X.to_numpy()
    y = data.iloc[:,1]
    y = y.to_numpy()
    
    #print(Y)
    
    if options.test: # Using external test
        data = pd.read_csv(options.test, sep=options.delim, header=options.header)
        X_test = data.iloc[:,2:]
        X_test = X_test.to_numpy()
        X_train = X
        y_train = y
        ID = data.iloc[:,0]
        ID = ID.to_numpy()
    
        # Fit model
        
        model = lgbm_model()
        model.fit(X_train, y_train)
    
        predictions = model.predict(X_test)
        
        # Save predictions to file
        with open(outfile, "w") as f:
            for n, pred in enumerate(predictions):
                new_line = "%s %s\n" % (ID[n], pred)
                f.write(new_line)
    
    
    else: # Using Kfold internal validation
        
        kf = KFold(n_splits=folds, random_state=random_seed, shuffle=True)
        
        mse_list = []
        rmse_list = []
        r2_list = []
            
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            model = lgbm_model()
                                  
            model.fit(X[train_index], y[train_index])
    
            predictions = model.predict(X[test_index])
            
            r2_list.append(r2_score(y[test_index], predictions))
            mse_list.append(mean_squared_error(y[test_index], predictions))
            rmse_list.append(root_mean_squared_error(y[test_index], predictions))
        
        # Output model evaluation
        print('R2: ', np.mean(r2_list))
        print('MSE: ', np.mean(mse_list))
        print('RMSE: ', np.mean(rmse_list))
    
