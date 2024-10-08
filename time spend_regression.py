# import the necessary files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from xgboost import XGBRegressor
import xgboost
from sklearn.metrics import mean_squared_error

# import the training and testing data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print('Training data: \n',train.head())
print('Testing data: \n',test.head())
print(train.info())
print(test.info())

# spliting inputs and outputs of training data
X_train, y_train = train.drop('y', axis=1), train['y']

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# Understanding the dataset
print(X_train.describe())

# Sepearting variables with categorical and non categorical values 
X_train_OB = X_train.select_dtypes(include=['object'])
X_train = X_train.select_dtypes(exclude=['object'])
test_OB = test.select_dtypes(include=['object'])
test = test.select_dtypes(exclude=['object'])

print('Training data without categorical values: \n',X_train.head())
print('Training data with categorical values: \n' ,X_train_OB.head())
print('Testing data without categorical values: \n' ,test.head())
print('Testing data with categorical values: \n' ,test_OB.head())


# check if any columns the variance is equal to zero in numerical training dataset
zero_variances_columns = [col for col in X_train.columns if X_train[col].var() == 0]
print('Zero variance columns: \n',zero_variances_columns)

# drop variables with 0 variance #
X_train = X_train.drop(zero_variances_columns, axis=1)
test = test.drop(zero_variances_columns, axis=1)
print('Training data shape after dropping zero variance columns: \n',X_train.shape)
print('Testing data shape after dropping zero variance columns: \n' ,test.shape)

# drop vaibles with N/A values
X_train = X_train.dropna(axis=0)
test = test.dropna(axis=0)

# standardizing the numerical training datset
standardize = preprocessing.MinMaxScaler() # using Sklearn standarsizing function to create an object stadardize
scaled_X = standardize.fit_transform(X_train)  # Fit methods calculates the stadardization of input data set, then transform methods applies it to it.
scaled_X = pd.DataFrame(scaled_X, columns=X_train.columns)
scaled_test = standardize.fit_transform(test)
scaled_test = pd.DataFrame(scaled_test, columns=test.columns)
print("The new scaled input datsset is: \n",scaled_X)

# apllying a One hot encoding to categorical training data
OHE = OneHotEncoder(handle_unknown = "ignore")
X_train_OB = OHE.fit_transform(X_train_OB).toarray()
col = OHE.get_feature_names_out()
col = np.array(col).ravel()
X_train_OHE  =pd.DataFrame(X_train_OB, columns=col)
print('Label encoded training data: \n',X_train_OHE)

# Doing the same on testing dataset
test_OB = OHE.fit_transform(test_OB).toarray()
col = OHE.get_feature_names_out()
col = np.array(col).ravel()
test_OHE  =pd.DataFrame(test_OB, columns=col)
print('Label encoded testing data: \n',test_OHE)

# Combining categorical and numerical data into one data frame of training data
X_train_complete = pd.concat([X_train_OHE, scaled_X], axis=1)
print('Combined training data: \n',X_train_complete)

X_test_complete = pd.concat([test_OHE, scaled_test], axis=1)
print('Combined testing data: \n',X_test_complete)

# Applying PCA
# applying dimension reduction 
pca = PCA()
x_train_pca = pca.fit(X_train_complete)
# finding the number of components needed to retain atleat 95% of the variance
figure = plt.figure(figsize=(10,5))
cumsum = np.cumsum(x_train_pca.explained_variance_ratio_)
plt.plot(cumsum, marker='o')
plt.title('Cumulative explained variance')
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()
# No.  of dimensions needed 
d = np.argmax(cumsum >= 0.95) + 1
print('\n Number of dimendsion needed to retain 95% of the variance: ',d)

# Since the features are too many we dimensionally reduce the features to a number of components that will retain atleat 95% of the varaince 
pca = PCA(n_components=d)
x_train_pca = pca.fit_transform(X_train_complete)
x_train_pca = pd.DataFrame(x_train_pca)

x_test_pca = pca.fit_transform(X_test_complete)
x_test_pca = pd.DataFrame(x_test_pca)
print("\n Feature dimensions reduced from  ",X_train_complete.shape," to ",x_train_pca.shape) # we print the no. of priciple components we get when 99% of caricae is retained. 

print('Input after PCA: \n',x_train_pca.head())
figure = plt.figure(figsize=(10,5))
plt.plot(cumsum, marker='o')
plt.show()
# Spliting the training dataframe into training and validation
X_train, X_val, y_train, y_val = train_test_split(x_train_pca, y_train, test_size=0.2, random_state=42)


# Creating XG-boost regressor model
model = XGBRegressor(n_estimators=10000, learning_rate=0.01, n_jobs=-1, booster='gbtree', device= 'cuda', 
                     callbacks=[xgboost.callback.EarlyStopping(rounds=1000, save_best=True, metric_name='rmse', min_delta=0.001, data_name="validation_0" )])

# Fitting the model to the training data
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

# testing the model with R2 score as accuracy
score = model.score(X_val, y_val)
print("\n The score of the model on the validation data is:", score*100,"% Accuracy")
print("\n The RMSE of the model on the validation data is:", np.sqrt(mean_squared_error(y_val, model.predict(X_val))))

# prediction fro validation data
predict = model.predict(X_val) 
# next we use pandas to create a datafram to compare the predicted values Vs the actual values and we print the result
compare=pd.DataFrame(data={"Predicted Values":predict}) # crearting data frame using with predicted values
compare.insert(loc=0, column='Actural Value', value=y_val.values ) # adding the acutal values at 1st loaction of the dataframe by changing the test dataset to numpy array
# we remove the difference between the actual vs predicted
compare['Difference'] = compare['Actural Value'] - compare['Predicted Values']
print(" The predicted values for validation data are: \n",compare)
print('Average difference between actual and predicted values for validation data:',compare['Difference'].mean())

# predicting testing data
test_predict = model.predict(x_test_pca)
# next we use pandas to create a datafram to compare the predicted values Vs the actual values and we print the result
test_predict_values=pd.DataFrame(data={"Predicted Values":test_predict}) # crearting data frame using with predicted values
print("\n The predicted values for the testing dataare: \n",test_predict_values) 