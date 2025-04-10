import numpy as np
import pandas as pd
from keras.src import regularizers
from keras import backend
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn import metrics
import tensorflow
from sklearn.metrics import mean_squared_error
def rmse(y_obs, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_obs), axis=-1))
def mse(y_obs, y_pred):
    return backend.mean(backend.square(y_pred - y_obs), axis=-1)
def r_square(y_obs, y_pred):
    SS_res =  backend.sum(backend.square(y_obs - y_pred))
    SS_tot = backend.sum(backend.square(y_obs - backend.mean(y_obs)))
    return (1 - SS_res/(SS_tot + backend.epsilon()))
def mean_absolute_percentage_error(y_obs, y_pred):
    y_obs, y_pred = np.array(y_obs), np.array(y_pred)
    y_obs=y_obs.reshape(-1,1)
    #y_obs, y_pred =check_array(y_obs, y_pred)
    return  np.mean(np.abs((y_obs - y_pred) / y_obs)) * 100
def mean_absolute_percentage_error2(y_obs, y_pred): #for when the MAPE doesnt need reshaping
    y_obs, y_pred = np.array(y_obs), np.array(y_pred)
    #y_obs=y_obs.reshape(-1,1)
    #y_obs, y_pred =check_array(y_obs, y_pred)
    return  np.mean(np.abs((y_obs - y_pred) / y_obs)) * 100
model = Sequential()
model.add(Dense(units = 512, activation = 'relu', input_dim = 15 , name='scratch_dense_1', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(units = 256, activation = 'relu', name='scratch_dense_2', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(units = 256, activation = 'relu', name='scratch_dense_3', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(units = 128, activation = 'relu', name='scratch_dense_4', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(units = 16, activation = 'relu', name='scratch_dense_5', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(units = 1, name='scratch_dense_6'))
adam=optimizers.Adam(lr=0.0003,beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer = adam, loss = 'mean_squared_error',metrics=['mean_squared_error', rmse,r_square] )
dataset = pd.read_csv("D:\double modle\droplet.csv",encoding="gbk")#Please select your dataset path
X = dataset.iloc[:,0:15]
Y = dataset.iloc[:,-1]
from sklearn.model_selection import train_test_split, GridSearchCV

X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.2,
                                                    random_state=0)
result = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size = 32, epochs = 3000)##将训练数据在模型中训练一定次数，返回loss和测量指标
y_pred_train = model.predict(X_train)
y_pred= model.predict(X_test)
##作RMSE训练历史图
plt.plot(result.history['rmse'])
plt.plot(result.history['val_rmse'])
plt.title('rmse')
plt.ylabel('rmse')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
##作MSE训练历史图
plt.plot(result.history['mean_squared_error'])
plt.plot(result.history['val_mean_squared_error'])
plt.title('loss function')
plt.ylabel('mean squared error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
train_erros_sum = []
test_erros_sum = []
train_mse_sum = []
test_mse_sum = []
results=[]
y_pred=np.array(y_pred)
y_pred=y_pred.flatten()
y_pred_train=np.array(y_pred_train)
y_pred_train=y_pred_train.flatten()
test_error=abs(y_pred-Y_test)/Y_test
test_mse=mean_squared_error(y_pred,Y_test)
train_error=abs(y_pred_train-Y_train)/Y_train
train_mse=mean_squared_error(y_pred_train,Y_train)
if np.mean(test_error)<0.5 and np.mean(train_error)<0.5:
    test_erros_sum.append(np.mean(test_error))
    train_erros_sum.append(np.mean(train_error))
    test_mse_sum.append(test_mse)
    train_mse_sum.append(train_mse)
results.append([train_erros_sum,train_mse_sum,test_erros_sum,test_mse_sum])
results = pd.DataFrame(results,
                               columns=['train_error', 'train_mse', 'test_error',
                                         'test_mse'])
results.to_excel("D:\double modle\inner diameter_results.xlsx")
Y_test=list(Y_test)
test_val=pd.DataFrame(Y_test,columns=['val'])
test_pred=pd.DataFrame(y_pred,columns=['pred'])
test_results=pd.concat([test_val,test_pred],axis=1)
test_results.to_excel("D:\double modle\inner diameter_results - 2.xlsx")
model.save('D:\double modle5.h5')