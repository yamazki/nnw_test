import csv 
import random
import numpy as np

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
  return np.random.randn(input_size, hidden_size)
      
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

def predict(network, iris_data):
  pass

# ニューラルネットワークの作成
# pathを引数とした場合,学習済みのネットワークを読み込む
# なければ適当に作成
def init_network(path=None):
  if(path == None):
    network = {}
    network['W1'] = generate_weight(3, 100) 
    network['W2'] = generate_weight(100, 100) 
    network['W3'] = generate_weight(100, 3) 
    network['B1'] = generate_bias(100)
    network['B2'] = generate_bias(100)
    network['B3'] = generate_bias(3)
    return network
  return 
  
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
  return 0.5 * np.sum((y - t) ** 2)
  
# 損失関数(交差エントロピー誤差)
def cross_entropy_error(y, t):
  if y.ndim == 1:
    t = t.reahspe(1, t.size)
    y = y.reahspe(1, y.size)
  batch_size = y.shape[0]
  delta = 1e-7 #np.log(y)が無限大になるのを防ぐ
  return - np.sum(t * np.log(y + delta)) / batch_size
  
iris_data = get_iris_data('./data.csv')
B1 = generate_bias(100)
B2 = generate_bias(100)
B3 = generate_bias(100)

train_iris_data, test_iris_data = split_iris_data(iris_data, 0.8)

## train_iris_data_features = list(map(lambda iris: iris["feature"], train_iris_data))
## train_iris_data_labels = list(map(lambda iris: iris["label"], train_iris_data))
## test_iris_data_features = list(map(lambda iris: iris["feature"], test_iris_data))
## test_iris_data_labels = list(map(lambda iris: iris["label"], test_iris_data))

epochs = 100  
batch_size = 10


network = init_network()
W1, W2, W3 = network['W1'], network['W2'], network['W3']
B1, B2, B3 = network['B1'], network['B2'], network['B3']

for i in range(epochs):
  selected_iris_data_list = random.choices(train_iris_data, k=batch_size)
  iris_features = list(map(lambda iris: iris["feature"], selected_iris_data_list))
  iris_labels   = list(map(lambda iris: convert_number_to_array(iris["label"], 3), selected_iris_data_list))
  A1 = np.dot(iris_features, W1) + B1
  Z1 = sigmoid(A1)
  A2 = np.dot(Z1, W2) + B2
  Z2 = sigmoid(A2)
  A3 = np.dot(Z2, W3) + B3
  Z3 = sigmoid(A3)
  
  y = np.array(list(map(lambda output: soft_max(output), Z3)))
  print(y)
  print(iris_labels)
  print(mean_aquarde_error(y, iris_labels) / batch_size)
  print(cross_entropy_error(y, iris_labels))

  
  


# H1 = np.dot()





