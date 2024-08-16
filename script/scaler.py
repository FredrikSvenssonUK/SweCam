"""
Script for the Tox21 challenge.
https://github.com/FredrikSvenssonUK/SweCam/blob/main/LICENSE

Scale data and save scaler.

Using MinMaxScaler.
"""


__author__ = "Ulf Norinder, Fredrik Svensson"
__date__ = "15/08/2024"


### Imports ###
import joblib
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd


### Config ###
train_file = "data/tox24_challenge_train_mordred.csv"
test_file = "data/tox24_challenge_test_mordred.csv"
leaderboard_file = "data/tox24_challenge_leaderboard_mordred.csv"
delim = "\t"
setheader = None

scaler_out = "scaler"
train_file_out = "data/tox24_challenge_train_mordred_scaled.csv"
test_file_out = "data/tox24_challenge_test_mordred_scaled.csv"
leaderboard_file_out = "data/tox24_challenge_leaderboard_mordred_scaled.csv"


#### Main ####

# Load data
data = pd.read_csv(train_file, sep=delim, header=setheader)
X = data.iloc[:,2:]
X = X.to_numpy()

# Fit and scale training data
scaler_X = MinMaxScaler()
X = scaler_X.fit_transform(X)
data.iloc[:,2:] = X
data.to_csv(train_file_out, header=False, index = False, sep=delim)

# Scale test data
data = pd.read_csv(test_file, sep=delim, header=setheader)
X = data.iloc[:,2:]
X = X.to_numpy()
X = scaler_X.transform(X)
data.iloc[:,2:] = X
data.to_csv(test_file_out, header=False, index = False, sep=delim)

# Save the scaler
joblib.dump(scaler_X, scaler_out)
