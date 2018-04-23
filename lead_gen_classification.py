import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.utils import np_utils # For y values
import time

dataset = pd.read_csv('./lead_gen.csv')

## manipuldated data
## print(dataset.head(20))
## print('total data : ', dataset.shape)
## print('distinct value of source')
## print(dataset['source'].unique())
## print('distinct value of country')
## print(dataset['country'].unique())

## Change categorial data to numerical data
dataset["source"] = pd.Categorical(dataset['source'], dataset['source'].unique())
dataset["source"]  = dataset["source"].cat.rename_categories([1,2,3,4])
dataset["country"] = pd.Categorical(dataset['country'], dataset['country'].unique())
dataset["country"]  = dataset["country"].cat.rename_categories([1,2,3])
## split train test data

train= dataset.sample(frac=0.6,random_state=200)
test= dataset.drop(train.index)
print("train data size: ", train.shape)
print("test data size: ",test.shape)

## split input , output data for train
X = train.iloc[:, :-1].values
Y = train.iloc[:, -1:].values


# Get dimensions of input and output
dimof_input_train = X.shape[1]
dimof_output_train = np.max(Y) + 1

# Set y categorical
Y = np_utils.to_categorical(Y, dimof_output_train)

# Set constants
dimof_middle = 100
# Set model
model = Sequential()
model.add(Dense(dimof_middle, input_dim = dimof_input_train, activation='tanh'))
model.add(Dense(dimof_middle, activation='tanh'))
model.add(Dense(dimof_middle, activation='tanh'))
model.add(Dense(dimof_output_train, activation='softmax'))
model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
start_time = time.time()
model.fit(X, Y,epochs= 100, batch_size=128, verbose = 2)
elapsed_time = time.time() - start_time
print("run time : ", elapsed_time)
loss, accuracy = model.evaluate(X, Y, verbose=0, batch_size=128)
print('loss train: ', loss)
print('accuracy train: ', accuracy)

X_TEST = test.iloc[:, :-1].values
Y_TEST = test.iloc[:, -1:].values
Y_TEST = np_utils.to_categorical(Y_TEST, dimof_output_train)

loss, accuracy = model.evaluate(X_TEST, Y_TEST, verbose=0, batch_size=128)
print('loss test: ', loss)
print('accuracy test: ', accuracy)
None



# Get dimensions of input and output
## prediction

