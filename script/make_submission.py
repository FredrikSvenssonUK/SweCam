"""
Script for the Tox21 challenge.
https://github.com/FredrikSvenssonUK/SweCam/blob/main/LICENSE

Take the output from a predictor and format for submission.

Missing values are inserted as the mean prediction of the total dataset.
"""


__author__ = "Ulf Norinder, Fredrik Svensson"
__date__ = "16/08/24"


### Imports ###
import numpy as np


### Config ###
infile = "lgbm_predictions.csv"
outfile = "submission.csv"


### Main ###
data = []
IDs = []
with open(infile) as f:
    for line in f:
        line = line.strip()
        ID, value = line.split()
        IDs.append(int(ID))
        data.append(float(value))

mean_data = np.mean(data)

outfile = open(outfile, "w")
header = "Predictions\n"
outfile.write(header)

i = 0
missing = 0
missing_list = []
for n, line in enumerate(data):
    print(i+1, n, IDs[n])
    ID = IDs[n]
    while ID != (i+1):
        new_line = "%s\n" % mean_data
        outfile.write(new_line)
        i+=1
        missing+=1
        missing_list.append(i)
    
    new_line = "%s\n" % line
    outfile.write(new_line)
    i+=1
    
outfile.close()
print("")
print("Estimated %s instances" % missing)
print(missing_list)
