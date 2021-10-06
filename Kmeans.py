import numpy as np # used for handling numbers
import pandas as pd # used for handling the dataset
from sklearn.preprocessing import Normalizer

from sklearn.impute import SimpleImputer # used for handling missing data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # used for encoding categorical data
from sklearn.model_selection import train_test_split # used for splitting training and testing data
from sklearn.preprocessing import StandardScaler # used for feature scaling


dataset_filename = 'kddtrain.csv'
dataset = pd.read_csv(dataset_filename)
print(dataset)

#dataset = pd.read_csv(dataset_filename, sep=',', decimal='.', header=None)



# Splitting the attributes into independent and dependent attributes
X = dataset.iloc[:, 0:42].values # attributes to determine dependent variable / Class
Y = dataset.iloc[:, 42].values # dependent variable / Class


print(X)
print(Y)


scaler = Normalizer().fit(X)
trainX = scaler.transform(X)
print(trainX)


