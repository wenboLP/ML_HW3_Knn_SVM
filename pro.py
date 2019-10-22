import numpy as np
np.random.seed(37)
import random
import pandas as pd

from sklearn.svm import SVC
# Att: You're not allowed to use modules other than SVC in sklearn, i.e., model_selection.

# Dataset information
# the column names (names of the features) in the data files
# you can use this information to preprocess the features
col_names_x = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
             'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
             'hours-per-week', 'native-country']
col_names_y = ['label']

numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
                  'hours-per-week']
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                    'race', 'sex', 'native-country']


def load_data(csv_file_path):
    # your code here
    # csv_file_path = "salary.labeled.csv"  # for test ~~~~~~~~~~~~~
    data = pd.read_csv(csv_file_path, names=col_names_x+col_names_y,  sep=',', na_values='?')

    data_num = data.loc[:, numerical_cols]
    data_num_norm = (data_num - data_num.min()) / (data_num.max() - data_num.min())
    # data_num_norm.columns

    data_catg = data.loc[:, categorical_cols]
    # data_catg.columns
    data_catg_encode = pd.get_dummies(data_catg) 
    # data_catg_encode.columns
    
    data_processed = pd.concat([data_num_norm, data_catg_encode], axis=1)
    # data_num_norm.columns 
    # data_catg_encode.columns
    # data_processed.columns   
    # a = list(data_num_norm.columns) + list(data_catg_encode.columns)
    # b =  list(data_processed.columns)
    # a==b
    x = data_processed
    y = data.loc[:,'label']
    y [y == ' >50K' ] = 1
    y [y == ' <=50K' ] = 0
    # data.loc[:,'label']  # ？？？
    return x, y


training_csv = "salary.labeled.csv"  # for test ~~~~~~~~~~~~~
x_train, y_train = load_data(training_csv)
x_val = x_train.values
y_val = y_train.values
y_val=y_val.astype('int')

my_nodel  = SVC( C=1.0, degree=1)
my_nodel.fit(x_val, y_val)





























