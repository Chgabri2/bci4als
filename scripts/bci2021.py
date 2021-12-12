import pickle
import os
import numpy as np
import pandas as pd

objectRep = open("C:\\Users\\asus\\OneDrive\\BSC_brain_math\\year_c\\Yearly\\BCI\\bci4als\\recordings\\adi\\8\\trials.pickle", "rb")
file = pickle.load(objectRep)
all_data = np.zeros([len(file), 120, 13])
print(file)
#
# for i in range(0,13):

