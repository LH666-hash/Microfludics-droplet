import numpy as np
import pandas as pd
from keras.src import regularizers
from keras import backend
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from keras.layers import Input, Dense
from keras import optimizers
from sklearn import metrics
import tensorflow
from sklearn.metrics import mean_squared_error

# 定义自定义评估函数
def rmse(y_obs, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_obs), axis=-1))

def mse(y_obs, y_pred):
    return backend.mean(backend.square(y_pred - y_obs), axis=-1)

def r_square(y_obs, y_pred):
    SS_res = backend.sum(backend.square(y_obs - y_pred))
    SS_tot = backend.sum(backend.square(y_obs - backend.mean(y_obs)))
    return (1 - SS_res / (SS_tot + backend.epsilon()))

def mean_absolute_percentage_error(y_obs, y_pred):
    y_obs, y_pred = np.array(y_obs), np.array(y_pred)
    y_obs = y_obs.reshape(-1, 1)
    return np.mean(np.abs((y_obs - y_pred) / y_obs)) * 100

def mean_absolute_percentage_error2(y_obs, y_pred):
    y_obs, y_pred = np.array(y_obs), np.array(y_pred)
    return np.mean(np.abs((y_obs - y_pred) / y_obs)) * 100

# 定义输入层
input_layer = Input(shape=(7,), name='input')

# 定义共享层
x = Dense(units=512, activation='relu', name='shared_dense_1', kernel_regularizer=regularizers.l2(0.001))(input_layer)
x = Dense(units=256, activation='relu', name='shared_dense_2', kernel_regularizer=regularizers.l2(0.001))(x)
x = Dense(units=256, activation='relu', name='shared_dense_3', kernel_regularizer=regularizers.l2(0.001))(x)

# 定义两个输出层
output1 = Dense(units=1, name='output1')(x)
output2 = Dense(units=1, name='output2')(x)

# 定义模型
model = Model(inputs=input_layer, outputs=[output1, output2])

# 编译模型
adam = optimizers.Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer=adam,
              loss={'output1': 'mean_squared_error', 'output2': 'mean_squared_error'},
              metrics={'output1': ['mean_squared_error', rmse, r_square],
                       'output2': ['mean_squared_error', rmse, r_square]})

# 读取数据
dataset = pd.read_csv("D:\py\droplet.csv", encoding="gbk")
X = dataset.iloc[:, 0:7]
Y1 = dataset.iloc[:, -2]  # 假设第一个输出是最后一列
Y2 = dataset.iloc[:, -1]  # 假设第二个输出是倒数第二列

# 划分训练集和测试集
from sklearn.model_selection import train_test_split

X_train, X_test, Y1_train, Y1_test, Y2_train, Y2_test = train_test_split(X, Y1, Y2,
                                                                          test_size=0.2,
                                                                          random_state=0)

# 训练模型
result = model.fit(X_train, {'output1': Y1_train, 'output2': Y2_train},
                   validation_data=(X_test, {'output1': Y1_test, 'output2': Y2_test}),
                   batch_size=32, epochs=3000)

# 预测
y_pred_train1, y_pred_train2 = model.predict(X_train)
y_pred1, y_pred2 = model.predict(X_test)

# 作RMSE训练历史图
plt.plot(result.history['output1_rmse'])
plt.plot(result.history['val_output1_rmse'])
plt.title('output1 rmse')
plt.ylabel('rmse')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(result.history['output2_rmse'])
plt.plot(result.history['val_output2_rmse'])
plt.title('output2 rmse')
plt.ylabel('rmse')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# 作MSE训练历史图
plt.plot(result.history['output1_mean_squared_error'])
plt.plot(result.history['val_output1_mean_squared_error'])
plt.title('output1 loss function')
plt.ylabel('mean squared error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

plt.plot(result.history['output2_mean_squared_error'])
plt.plot(result.history['val_output2_mean_squared_error'])
plt.title('output2 loss function')
plt.ylabel('mean squared error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# 计算误差
train_erros_sum = []
test_erros_sum = []
train_mse_sum = []
test_mse_sum = []
results = []

y_pred1 = np.array(y_pred1).flatten()
y_pred2 = np.array(y_pred2).flatten()
y_pred_train1 = np.array(y_pred_train1).flatten()
y_pred_train2 = np.array(y_pred_train2).flatten()

test_error1 = abs(y_pred1 - Y1_test) / Y1_test
test_mse1 = mean_squared_error(y_pred1, Y1_test)
train_error1 = abs(y_pred_train1 - Y1_train) / Y1_train
train_mse1 = mean_squared_error(y_pred_train1, Y1_train)

test_error2 = abs(y_pred2 - Y2_test) / Y2_test
test_mse2 = mean_squared_error(y_pred2, Y2_test)
train_error2 = abs(y_pred_train2 - Y2_train) / Y2_train
train_mse2 = mean_squared_error(y_pred_train2, Y2_train)

if np.mean(test_error1) < 0.5 and np.mean(train_error1) < 0.5:
    test_erros_sum.append(np.mean(test_error1))
    train_erros_sum.append(np.mean(train_error1))
    test_mse_sum.append(test_mse1)
    train_mse_sum.append(train_mse1)

if np.mean(test_error2) < 0.5 and np.mean(train_error2) < 0.5:
    test_erros_sum.append(np.mean(test_error2))
    train_erros_sum.append(np.mean(train_error2))
    test_mse_sum.append(test_mse2)
    train_mse_sum.append(train_mse2)

results.append([train_erros_sum, train_mse_sum, test_erros_sum, test_mse_sum])
results = pd.DataFrame(results, columns=['train_error', 'train_mse', 'test_error', 'test_mse'])
results.to_excel("D:\double_model\\results.xlsx")

# 保存预测结果
Y1_test = list(Y1_test)
Y2_test = list(Y2_test)
test_val1 = pd.DataFrame(Y1_test, columns=['val1'])
test_val2 = pd.DataFrame(Y2_test, columns=['val2'])
test_pred1 = pd.DataFrame(y_pred1, columns=['pred1'])
test_pred2 = pd.DataFrame(y_pred2, columns=['pred2'])

test_results1 = pd.concat([test_val1, test_pred1], axis=1)
test_results2 = pd.concat([test_val2, test_pred2], axis=1)

test_results1.to_excel("D:\double_model\\test_results1.xlsx")
test_results2.to_excel("D:\double_model\\test_results2.xlsx")

# 保存模型
model.save('D:\double_model.h5')
