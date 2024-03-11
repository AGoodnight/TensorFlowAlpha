import tensorflow as tf
from tensorflow import keras
from keras import layers

import matplotlib.pyplot as plt

import numpy as np
# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

import pandas as pd
import seaborn as sns

data = pd.read_csv('input/diabetes.csv')
column_names = np.array(data.columns)

train_dataset = data.sample(frac=0.8, random_state=0)
test_dataset = data.drop(train_dataset.index)

sns.pairplot(train_dataset[column_names], diag_kind='kde')
print(train_dataset.describe().transpose())

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('Outcome')
test_labels = test_features.pop('Outcome')

train_dataset.describe().transpose()[['mean', 'std']]

