def predict(test_csv, trained_model，x_train ):
    test_csv = "salary.2Predict.csv"  # for test ~~~~~~~~~~~~
    x_test, _ = load_data(test_csv)


    predictions = trained_model.predict(x_test)
    return predictions

x_train  # 107
x_train.columns
x_test  # 108
x_test.columns

miss_col = set(x_train.columns) - set(x_test.columns)
miss_col
for col in miss_col:
    x_test[col] = 0

extr_col = set(x_test.columns) - set(x_train.columns)
extr_col
x_test.drop(list(extr_col), axis = 1, inplace =  True  )

for i in range(len(x_test.columns)):
    if x_train.columns[i] != x_test.columns[i]:
        print(i)
