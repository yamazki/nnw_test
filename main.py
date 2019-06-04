import csv 
import random
import numpy as np
import pickle

# filepathに格納されているirisのデータを
# {"feature" : np.array(irisの特徴量), "iris_type" : irisの種類} リストで返す
def get_iris_data(filepath):
  with open(filepath) as f:
    reader = csv.reader(f)
    iris_list = []
    for row in reader:
      iris_feature = np.array(list(map(lambda x: (float(x)), row[0:3])))
      iris_type = int(row[4])
      iris = {"feature" : iris_feature, "label" : iris_type}
      iris_list.append(iris)
    f.close
    return iris_list
    
# iris_dataを訓練用とテスト用に分類,それらをreturn
# splitは0~1の値を取る
def split_iris_data(iris_data, split):
  if(split < 0 or split > 1): 
    raise ValueError("split must be bigger than 0 and smaller than 1")
  random.shuffle(iris_data)
  spliter = int(len(iris_data) * split)
  train_iris_data = iris_data[0:spliter]
  test_iris_data = iris_data[spliter+1:len(iris_data)-1]
  return train_iris_data, test_iris_data


# input_sizeは行数
# hidden_sizeは列数
def generate_weight(input_size, hidden_size):
  return  np.random.randn(input_size, hidden_size)
      
def generate_bias(hidden_size):
  return np.zeros(hidden_size)
      
def sigmoid(x):
  return 1 / (1 + np.exp(-x))
      
# ソフトマックス関数
def soft_max(a):
  c = np.max(a)
  exp_a = np.exp(a - c) #オーバーフロー対策
  sum_exp_a = np.sum(exp_a)
  return exp_a / sum_exp_a


# ニューラルネットワークの作成
# pathを引数とした場合,学習済みのネットワークを読み込む
# なければ適当に作成
def init_network(path=None):
  if(path == None):
    network = {}
    network['W1'] = generate_weight(3, 20) 
    network['W2'] = generate_weight(20, 20) 
    network['W3'] = generate_weight(20, 3) 
    network['B1'] = generate_bias(20)
    network['B2'] = generate_bias(20)
    network['B3'] = generate_bias(3)
    return network
  with open('./learning_data.pickle', mode='rb') as fo:
    network = pickle.load(fo)  
  return network
  
# ラベルの数字を配列にする{0, 1, 2}の中の1ならば [0, 1, 0]とする
def convert_number_to_array (number, array_size):
  array = []
  for i in range(array_size):
    if(i == number):
      array.append(1)
    else:
      array.append(0)
  return array
    
# 損失関数(二乗和誤差)
def mean_aquarde_error(y, t):
  if y.ndim == 1:
    t = t.reahspe(1, t.size)
    y = y.reahspe(1, y.size)
  batch_size = y.shape[0]
  return 0.5 * np.sum((y - t) ** 2) / batch_size
  
# 損失関数(交差エントロピー誤差)
def cross_entropy_error(y, t):
  if y.ndim == 1:
    t = t.reahspe(1, t.size)
    y = y.reahspe(1, y.size)
  batch_size = y.shape[0]
  delta = 1e-7 #np.log(y)が無限大マイナスになるのを防ぐ
  return - np.sum(t * np.log(y + delta)) / batch_size
  
def caliculate_accuracy(y, t):
  y = np.argmax(y, axis=1)
  t = np.argmax(t, axis=1)
  return float(np.sum(y == t) / float(y.shape[0]))
  
def loss(network, iris_features, iris_labels):
  return numerical_gradient(lambda net: cross_entropy_error(predict(net, iris_features), iris_labels), network)

# 勾配計算
def numerical_gradient(f, network):
  h = 1e-4
  grads = {}
  for key,value in network.items(): 
    grad = np.zeros_like(value)
    it = np.nditer(value, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
      idx = it.multi_index
      tmp_val = value[idx]
      value[idx] = tmp_val + h
      network[key] = value
      fxh1 = f(network)
      
      value[idx] = tmp_val - h
      network[key] = value
      fxh2 = f(network)
      
      grad[idx] = (fxh1 - fxh2) / (2 * h)
      value[idx] = tmp_val
      network[key] = value
      it.iternext()
    grads[key] = grad
  return grads
    
# 勾配法
def gradient_descent(f, init_x, lr=0.01, step_num=100):
  x = init_x
  for i in range(step_num):
    grad = numerical_gradient(f, x)
    x -= lr * grad
  return x
    
def predict(network, iris_features):
  W1, W2, W3 = network['W1'], network['W2'], network['W3']
  B1, B2, B3 = network['B1'], network['B2'], network['B3']
  A1 = np.dot(iris_features, W1) + B1
  Z1 = sigmoid(A1)
  A2 = np.dot(Z1, W2) + B2
  Z2 = sigmoid(A2)
  A3 = np.dot(Z2, W3) + B3
  Z3 = sigmoid(A3)
  y = np.array(list(map(lambda output: soft_max(output), Z3)))
  return y
  
if __name__ == '__main__':
  iris_data = get_iris_data('./data.csv')
  epochs = 1000 
  batch_size = 3
  leaning_rate = 0.1
  network = init_network()
  # network = init_network('./leaning_data.pickle')
  
  train_iris_data, test_iris_data = split_iris_data(iris_data, 0.8)
  for i in range(epochs):
    devided_train_iris_data_list = [train_iris_data[i::batch_size] for i in range(batch_size)]
    for devided_train_iris_data in devided_train_iris_data_list:
      train_iris_features = list(map(lambda iris: iris["feature"], devided_train_iris_data))
      train_iris_labels   = list(map(lambda iris: convert_number_to_array(iris["label"], 3), devided_train_iris_data))
      grads = loss(network, train_iris_features, train_iris_labels)
      for key in network.keys(): 
        network[key] -= leaning_rate * grads[key]
    test_iris_features = list(map(lambda iris: iris["feature"], test_iris_data))
    test_iris_labels   = list(map(lambda iris: convert_number_to_array(iris["label"], 3), test_iris_data))
    result = predict(network, test_iris_features)
    print("loss: ", end="")
    print(cross_entropy_error(result, test_iris_labels))
    print('accuracy: ', end="")
    print(caliculate_accuracy(result, test_iris_labels))

  with open('./learning_data.pickle', mode='wb') as fo:
    pickle.dump(network, fo)



