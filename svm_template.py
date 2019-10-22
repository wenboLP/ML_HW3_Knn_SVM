# Starting code for UVA CS 4501 ML- SVM

import numpy as np
np.random.seed(37)
import random
import pandas as pd
from time import time

from sklearn.svm import SVC

# Dataset information

col_names_x = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
             'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
             'hours-per-week', 'native-country']
col_names_y = ['label']

numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
                  'hours-per-week']
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                    'race', 'sex', 'native-country']


# 1. Data loading from file and pre-processing.
# Hint: Feel free to use some existing libraries for easier data pre-processing. 
# For example, as a start you can use one hot encoding for the categorical variables and normalization 
# for the continuous variables.
def load_data(csv_file_path):
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
    y [y == ' <=50K' ] = -1
    # data.loc[:,'label']  # ？？？
    return x, y



def fold (x,y, ith_f, nfolds):
    n = len(x)
    # nfolds = 3
    np.random.seed(nfolds) 
    idx_sample = np.random.choice(n, n, replace=False)
    n_per_dold = int(np.ceil(n/nfolds))
    idx_array = np.zeros(n)
    if ith_f == nfolds-1:
        idx_array[ idx_sample[ith_f * n_per_dold: ] ] = 1
    else:
        idx_array[ idx_sample[ith_f * n_per_dold: (ith_f+1)*n_per_dold ] ] = 1
    x_cvtrain = x[idx_array==0]
    y_cvtrain = y[idx_array==0]
    x_cvtest = x[idx_array==1]
    y_cvtest = y[idx_array==1]
    return x_cvtrain, y_cvtrain, x_cvtest, y_cvtest

def acc_calculate(y_pre, y_test):
    return np.sum(y_pre*y_test == 1) / len(y_test)

def cross_vali(x_val, y_val, c_val, gama_val, n_fold ):
    # input: train data (x_val, y_val), model parameter (c_val, gama_val,), fold num (n_fold)
    # output: the accuracy of this parameter, using built-in model SVC(C=c_val, gamma=gama_val) 
    acc_total = 0
    for ith in range(n_fold):   # ith=1
        x_cvtrain, y_cvtrain, x_cvtest, y_cvtest = fold (x_val,y_val, ith, n_fold)
        my_nodel  = SVC( C=c_val, gamma=gama_val)
        my_nodel.fit(x_cvtrain, y_cvtrain)
        # score = my_nodel.score(x_cvtest, y_cvtest)
        y_cvpredict = my_nodel.predict(x_cvtest)   
        score = acc_calculate(y_cvpredict, y_cvtest)
        # score = np.sum(y_cvpredict*y_cvtest == 1) / len(y_test)
        acc_total += score
    return acc_total/n_fold

# 2. Select best hyperparameter with cross validation and train model.
# Attention: Write your own hyper-parameter candidates.
def train_and_select_model(training_csv):
    # load data and preprocess from filename training_csv
    training_csv = "salary.labeled.csv"  # for test ~~~~~~~~~~~~~
    x_train, y_train = load_data(training_csv)
    x_val = x_train.values
    y_val = y_train.values
    y_val=y_val.astype('int')

    # 设置参数
    # 每个参数CV得到acc
        # 输入：参数，数据
        # 输出：acc
    n_fold = 3

    # # one para test ------------
    # x_cvtrain, y_cvtrain, x_cvtest, y_cvtest = fold (x_val,y_val, 1, n_fold)
    # my_nodel  = SVC( C=1.0)
    # my_nodel.fit(x_cvtrain, y_cvtrain)
    # # score = my_nodel.score(x_cvtest, y_cvtest)
    # y_cvpredict = my_nodel.predict(x_cvtest)   
    # score = acc_calculate(y_cvpredict, y_cvtest)
    # # 1.0  gama默认 0.83758
    # # --------------------------

    # c_start, c_end = -3, 5      #-5 , 15 [ 0.125,  0.25,  0.5 ,  1.,  2.,  4.,  8. , 16. 32.   ]
    # c_num = c_end - c_start + 1
    # c_grid = np.logspace(c_start, c_end, num =c_num, base=2)
    c_grid = [0.001, 0.005, 0.01, 0.1, 0.5, 1, 5, 10]

    # gama_start, gama_end = -7,2      #-15,3
    # gama_num = gama_end - gama_start + 1
    # gama_grid = np.logspace(gama_start, gama_end, num =gama_num/2, base=2)
    gama_grid = [0.1,  0.2,  0.4,  0.8,  1.6, 3.2, 6.4]

    c_grid = [1.0,]
    gama_grid = [0.01, ]

    m_c = len(c_grid)
    n_g = len(gama_grid)
    acc_mat = np.zeros( (m_c, n_g) )
    best_acc = 0
    best_c, best_g = 0,0
    print ('m_c =',m_c, ';    n_g =', n_g)
    
    for i in range(m_c):
        print('current i = ', i, ' C = ', c_grid[i])
        for j in range(n_g):
            print(' j = ' , j, '  gama = ', gama_grid[j])
            acc = cross_vali(x_val, y_val, c_grid[i], gama_grid[j], n_fold )
            acc_mat[i][j] =  c_grid[i]
            if acc > best_acc:
                print('best_acc = ', acc)
                best_acc = acc
                best_c = c_grid[i]
                best_g = gama_grid[j]

    best_model =  SVC( C=best_c, gamma=best_g)
    best_model.fit(x_val, y_val)
    best_score = best_acc
    # your code here
    # iterate over all hyperparameter configurations
    # perform 3 FOLD cross validation
    # print cv scores for every hyperparameter and include in pdf report
    # select best hyperparameter from cv scores, retrain model 
    return best_model, best_score

# predict for data in filename test_csv using trained model
def predict(test_csv, trained_model):
    test_csv = "salary.2Predict.csv"  # for test ~~~~~~~~~~~~
    x_test, _ = load_data(test_csv)
    x_test.drop(['native-country_ Holand-Netherlands'], axis = 1, inplace =  True  )
    x_test_val = x_test.values
    predictions = trained_model.predict(x_test_val)
    return predictions

# save predictions on test data in desired format 
def output_results(predictions):
    with open('predictions.txt', 'w') as f:
        for pred in predictions:
            if pred == 0:
                f.write('<=50K\n')
            else:
                f.write('>50K\n')

if __name__ == '__main__':
    training_csv = "salary.labeled.csv"
    testing_csv = "salary.2Predict.csv"
    # fill in train_and_select_model(training_csv) to 
    # return a trained model with best hyperparameter from 3-FOLD 
    # cross validation to select hyperparameters as well as cross validation score for best hyperparameter. 
    # hardcode hyperparameter configurations as part of train_and_select_model(training_csv)
    trained_model, cv_score = train_and_select_model(training_csv)

    print( "The best model was scored %.2f" % cv_score )
    # use trained SVC model to generate predictions
    predictions = predict(testing_csv, trained_model)
    # Don't archive the files or change the file names for the automated grading.
    # Do not shuffle the test dataset
    output_results(predictions)
    # 3. Upload your Python code, the predictions.txt as well as a report to Collab.