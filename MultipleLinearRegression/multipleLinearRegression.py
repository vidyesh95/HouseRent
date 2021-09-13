import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn import preprocessing, linear_model

# Load data
print('-' * 30)
print("IMPORTING DATA")
print('-' * 30)
data = pd.read_csv('C:\\Users\churi\PycharmProjects\HouseRent\MultipleLinearRegression\houses_to_rent.csv', sep=',')
'''
data = data[
    ['city', 'rooms', 'area', 'parking spaces', 'bathroom', 'floor', 'animal', 'furniture', 'hoa', 'rent amount',
     'property tax', 'fire insurance', 'total']]
'''
data = data[
    ['city', 'rooms', 'area', 'parking spaces', 'bathroom', 'floor', 'animal', 'furniture', 'fire insurance',
     'rent amount']]
print(data.head())

# todo hoa,propery tax and lastly total

# Process data
print('-' * 30)
print("PROCESSING DATA")
print('-' * 30)
'''
data['total'] = data['total'].map(lambda i: int(i[2:].replace(',', '')))

# data['hoa'] = data['hoa'].map(lambda i: int(i[2:].replace(',', '')))
# data['hoa'] = data['hoa'].apply(lambda i: i.replace('R', '').replace('$', '').replace(',', ''))
# data['hoa'] = data['hoa'].astype(float)  # cast back to appropriate type
'''
data['rent amount'] = data['rent amount'].map(lambda i: int(i[2:].replace(',', '')))
'''
# .replace({'$': '', ',': ''}, regex=True).map(lambda i: int(i[1:]))
# data['property tax'] = data['property tax'].map(lambda i: i[2:]).str.replace(',', '')
# data['property tax'] = data['property tax'].apply(lambda i: i.replace('R', '').replace('$', '').replace(',', ''))
# data['property tax'] = data['property tax'].to_numeric()
# data['property tax'] = data['property tax'].astype(float)  # cast back to appropriate type
'''
data['fire insurance'] = data['fire insurance'].map(lambda i: int(i[2:].replace(',', '')))

data['floor'] = data['floor'].replace('-', np.nan)

labelEncoder = preprocessing.LabelEncoder()
data['furniture'] = labelEncoder.fit_transform(data['furniture'])

data['animal'] = labelEncoder.fit_transform(data['animal'])

print(data.head())

print('-' * 30)
print("CHECKING NULL DATA")
print('-' * 30)
print(data.isnull().sum())

print('-' * 30)
print("PROCESSED DATA")
print('-' * 30)
data = data.dropna()
print(data.head())
print(data.isnull().sum())

# Split data
print('-' * 30)
print("SPLIT DATA")
print('-' * 30)
'''
x = np.array(data.drop(['total'], axis=1))  # 1 axis to drop
y = np.array(data['total'])
'''
x = np.array(data.drop(['rent amount'], axis=1))  # 1 axis to drop
y = np.array(data['rent amount'])

print('X', x.shape)
print('Y', y.shape)
xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(x, y, test_size=0.2, random_state=10)
print('XTrain', xTrain.shape)
print('XTest', xTest.shape)

# Training data
print('-' * 30)
print("TRAINING DATA")
print('-' * 30)
model = linear_model.LinearRegression()
model.fit(xTrain, yTrain)
accuracy = model.score(xTest, yTest)
print('Coefficients : ', model.coef_)
print('Intercept : ', model.intercept_)
print('Accuracy : ', round(accuracy * 100, 3), '%')

# Evaluation
print('-' * 30)
print("MANUAL TESTING")
print('-' * 30)
testVal = model.predict((xTest))
print(testVal.shape)
error = []
for i, testVal in enumerate(testVal):
    error.append(yTest[i] - testVal)
    print(f'Actual value:{yTest[i]} Prediction value:{int(testVal)} Error:{int(error[i])}')
