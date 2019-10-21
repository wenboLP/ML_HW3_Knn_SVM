# Starting code for UVA CS 4501 ML- SVM

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


# 1. Data loading from file and pre-processing.
# Hint: Feel free to use some existing libraries for easier data pre-processing. 
# For example, as a start you can use one hot encoding for the categorical variables and normalization 
# for the continuous variables.
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

# 2. Select best hyperparameter with cross validation and train model.
# Attention: Write your own hyper-parameter candidates.
def train_and_select_model(training_csv):
    # load data and preprocess from filename training_csv
    training_csv = "salary.labeled.csv"  # for test ~~~~~~~~~~~~~
    x_train, y_train = load_data(training_csv)
    x_val = x_train.values
    y_val = y_train.values
    y_val=y_val.astype('int')

# for i in range(len(y_val)):
#     if ( y_val[i]!=0 and y_val[i]!=1 ) :
#         print(i)
#         print(y_val[i])


    # hard code hyperparameter configurations, an example:
    param_set = [
                 {'kernel': 'rbf', 'C': 1, 'degree': 1},
                 {'kernel': 'rbf', 'C': 1, 'degree': 3},
                 {'kernel': 'rbf', 'C': 1, 'degree': 5},
                 {'kernel': 'rbf', 'C': 1, 'degree': 7},
    ]

    my_nodel  = SVC( C=1.0, degree=1)
    my_nodel.fit(x_val, y_val)

    

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

    predictions = trained_model.predict(x_test)
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