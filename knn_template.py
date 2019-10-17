# Starting code for UVA CS 4501 Machine Learning- KNN

__author__ = '**'
import numpy as np
np.random.seed(37)
# for plot
import matplotlib.pyplot as plt
#more imports
from sklearn.neighbors import KNeighborsClassifier
## the only purpose of the above import is in case that you want to compare your knn with sklearn knn


# Load file into np arrays
# x is the features
# y is the labels
def read_file(file):
    data = np.loadtxt(file, skiprows=1)
    np.random.shuffle(data)
    x = data[:, :-1]
    y = data[:, -1].astype(int)
    return x, y

# 2. Generate the i-th fold of k fold validation
# Input:
# x is an np array for training data
# y is an np array for labels
# i is an int indicating current fold
# nfolds is the total number of cross validation folds
def fold(x, y, i, nfolds):
    # your code
    n = len(x)
    np.random.seed(nfolds)  # use the nfolds as seed 
    idx_sample = np.random.choice(n, n, replace=False)
    n_per_dold = int(np.ceil(n/nfolds))
    idx_array = np.zeros(n)
    if i>= nfolds:
        print("Wrong! i should be less than nfolds")
    if i == nfolds-1:
        idx_array[ idx_sample[i*n_per_dold: ] ] = 1
    else:
        idx_array[ idx_sample[i*n_per_dold: (i+1)*n_per_dold ] ] = 1
    x_train = x[idx_array==0]
    y_train = y[idx_array==0]
    x_test = x[idx_array==1]
    y_test = y[idx_array==1]
    return x_train, y_train, x_test, y_test

# 3. Classify each testing points based on the training points
# Input
# x_train: a numpy array of training data 
# x_test: a numpy array
# k: the number of neighbors to take into account when predicting the label
# Output
# y_predict: a numpy array 
def classify(x_train, y_train, x_test, k):
    # your code
    # Euclidean distance as the measurement of distance in KNN
    n_t = len(x_test)
    y_predict = np.zeros( n_t )
    for i in range(n_t):
        y_predict[i] = knn_vote(x_train, y_train, x_test[i], k )
    return y_predict

def knn_vote(x, y, aim, k_vote):
    # input x:all data, y:all label,  aim: the data to be classify
    # for y to be two classes and labeled as 0 or 1 
    # output will be 0 or 1 
    dis = np.sum( (x - aim)**2, axis = 1)
    idx = np.argsort(dis)[0:k_vote]   # equal value?
    vote = np.sum(y[idx])
    if vote>(k_vote/2):
        label_aim = 1
    else:
        label_aim = 0   
    return label_aim

# 4. Calculate accuracy by comaring with true labels
# Input
# y_predict is a numpy array of 1s and 0s for the class prediction
# y is a numpy array of 1s and 0s for the true class label
def calc_accuracy(y_predict, y):
    # your code
    flag = y_predict+y  # diff label will get 1 
    acc = 1 - ( np.sum( flag[flag==1] ) / len(y) )
    return acc

# 5. Draw the bar plot of k vs. accuracy
# klist: a list of values of ks
# accuracy_list: a list of accuracies
def barplot(klist, accuracy_list):
    # your code
    # use matplot lib to generate bar plot with K on x axis and cross validation accuracy on y-axis
    plt.bar(klist, accuracy_list, label= 'k vs accuracy')
    plt.legend()
    plt.show()
    return

# 1. Find the best K
def findBestK(x, y, klist, nfolds):
    kbest = 0
    best_acc = 0
    accuracy_list = []
    for k in klist:
        print(k)
        # your code here
        # to get nfolds cross validation accuracy for k neighbors
        # implement fold(x, y, i, nfolds),classify(x_train, y_train, x_test, k) and calc_accuracy(y_predict, y)

        acc_array_folds = np.zeros(nfolds)
        for i in range(nfolds):
            print(i)
            x_train, y_train, x_test, y_test = fold(x, y, i, nfolds)
            y_predict = classify(x_train, y_train, x_test, k)
            acc_array_folds[i] = calc_accuracy(y_predict, y_test)    
            print(acc_array_folds[i])
        accuracy = np.mean(acc_array_folds)  # CROSS VALIDATION accuracy for k neighbors
        if accuracy > best_acc:
            kbest = k
            best_acc = accuracy
        accuracy_list.append(accuracy)
        print(k, accuracy)
    # plot cross validation error for each k : implement function barplot(klist, accuracy_list)
    barplot(klist, accuracy_list)
    return kbest


if __name__ == "__main__":
    filename = "Movie_Review_Data.txt"
    # read data
    x, y = read_file(filename)
    nfolds = 4
    klist = [3, 5, 7, 9, 11, 13]
    # Implementation covers two tasks, both part of findBestK function
    # Task 1 : implement kNN classifier for a given x,y,k 
    # Task 2 : implement 4 fold cross validation to select best k from klist

    # nfold CV
    x_train, y_train, x_test, y_test = fold(x, y, 1, nfolds)

    # knn 
    # y_predict = classify(x_train, y_train, x_test, k):
    k_nei = 13
    y_predict = classify(x_train, y_train, x_test, k_nei)
    acc = calc_accuracy(y_predict, y_test)
    acc
     
    bestk = findBestK(x, y, klist, nfolds)
    # report best k, and accuracy, discuss why some k work better than others


# neigh = KNeighborsClassifier(n_neighbors=k_nei)
# neigh.fit(x_train, y_train) 
# y_predict_in = neigh.predict(x_test)
# acc = calc_accuracy(y_predict_in, y_test)
# acc
# acc_com = calc_accuracy(y_predict_in, y_predict)
# acc_com 
