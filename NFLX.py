import numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


netflix = pd.read_csv('NFLX.csv')
print(netflix.shape)
print(netflix.head())
# netflix=netflix[['Close']]
# print(netflix.head())
#
# # fix random seed for reproducibility
# np.random.seed(7)
# netflix = netflix.values
# netflix = netflix.astype('float32')
#
# # normalize the dataset
# scaler = MinMaxScaler(feature_range=(0, 1))
# netflix = scaler.fit_transform(netflix)
#
# # split into train and test sets
# train_size = int(len(netflix) * 0.70)
# test_size = len(netflix) - train_size
# train, test = netflix[0:train_size,:], \
#               netflix[train_size:len(netflix),:]
# print(len(train), len(test))
#
# #create look back function
# def create_netflix(netflix, look_back=1):
#     # convert an array of values into a dataset matrix
#     dataX, dataY = [], []
#     for i in range(len(netflix)-look_back-1):
#         a = netflix[i:(i+look_back), 0]
#         dataX.append(a)
#         dataY.append(netflix[i + look_back, 0])
#     return numpy.array(dataX), numpy.array(dataY)
# # fix random seed for reproducibility
# numpy.random.seed(7)
#
# # reshape into X=t and Y=t+1
# look_back = 1
# trainX, trainY = create_netflix(train, look_back)
# testX, testY = create_netflix(test, look_back)
#
# # reshape input to be [samples, time steps, features]
# trainX = numpy.reshape(trainX, (trainX.shape[0], 1,
#                                 trainX.shape[1]))
# testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
#
# # create and fit the LSTM network
# model = Sequential()
# model.add(LSTM(5, input_shape=(1, look_back)))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
#
# #predicting and inverse transforming the predictions
# # make predictions
# trainPredict = model.predict(trainX)
# testPredict = model.predict(testX)
# #invert predictions
# trainPredict = scaler.inverse_transform(trainPredict)
# trainY = scaler.inverse_transform([trainY])
# testPredict = scaler.inverse_transform(testPredict)
# testY = scaler.inverse_transform([testY])
# # calculate root mean squared error
# trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# print('Train Score: %.2f RMSE' % (trainScore))
# testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# print('Test Score: %.2f RMSE' % (testScore))
#
# # shift train predictions for plotting
# trainPredictPlot = numpy.empty_like(netflix)
# trainPredictPlot[:, :] = numpy.nan
# trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
#
# # shift test predictions for plotting
# testPredictPlot = numpy.empty_like(netflix)
# testPredictPlot[:, :] = numpy.nan
# testPredictPlot[len(trainPredict)+(look_back*2)+1:len(netflix)-1, :] = testPredict
#
# # plot baseline and predictions
# plt.plot(scaler.inverse_transform(netflix))
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
# plt.show()
# #
# #comparison of actual and predicted
# actual = np.append(trainY, testY)
# predicted = np.append(trainPredict, testPredict)
# result_df =pd.DataFrame()
# result_df['Actual_Y']= actual
# result_df['Predicted_Y']=predicted
# print(result_df.head())
#
#
#Filter data for period 21-04-2021 - 21-04-2022
netflix_x = netflix.loc[(netflix['Date'] >= '2021-04-01') & (netflix['Date'] < '2022-04-22')]
#print(netflix_x.head())
#netflix_x.to_csv('prev_data.csv', index=False)

netflix = pd.read_csv('prev_data.csv')
print(netflix.head())
netflix=netflix[['Close']]
print(netflix.head())


# fix random seed for reproducibility
np.random.seed(7)
netflix = netflix.values
netflix = netflix.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
netflix = scaler.fit_transform(netflix)

# split into train and test sets
train_size = int(len(netflix) * 0.70)
test_size = len(netflix) - train_size
train, test = netflix[0:train_size,:], \
              netflix[train_size:len(netflix),:]
print(len(train), len(test))

#create look back function
def create_netflix(netflix, look_back=1):
    # convert an array of values into a dataset matrix
    dataX, dataY = [], []
    for i in range(len(netflix)-look_back-1):
        a = netflix[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(netflix[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)
# fix random seed for reproducibility
numpy.random.seed(7)

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_netflix(train, look_back)
testX, testY = create_netflix(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1,
                                trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(5, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

#predicting and inverse transforming the predictions
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
#invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(netflix)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(netflix)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(netflix)-1, :] = testPredict

# plot baseline and predictions
plt.title('Predicted vs Actual for 12 months')
plt.plot(scaler.inverse_transform(netflix))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
#
#comparison of actual and predicted
actual = np.append(trainY, testY)
predicted = np.append(trainPredict, testPredict)
result_df =pd.DataFrame()
result_df['Actual_Y']= actual
result_df['Predicted_Y']=predicted
print(result_df.head())

