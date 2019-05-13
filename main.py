import csv 
import random
import numpy as np

# filepathに格納されているirisのデータを
# {"data" : np.array(irisの特徴量), "iris_type" : irisの種類} リストで返す
def get_iris_data(filepath):
  with open(filepath) as f:
    reader = csv.reader(f)
    return_list = []
    for row in reader:
      iris_data = np.array(list(map(lambda x: ([float(x)]), row[0:3])))
      iris_type = int(row[4])
      iris = {"data" : iris_data, "iris_type" : iris_type}
      return_list.append(iris)
    f.close
    return  return_list
    
# iris_dataを訓練用とテスト用に分類,それらをreturn
# splitは0~1の値を取る
def split_iris_data(iris_data, split):
  if(split < 0 or split > 1): 
    raise ValueError("split must be bigger than 0 and smaller than 1")
  random.shuffle(iris_data)
  spliter = int(len(iris_data) * split)
  train_iris_data = iris_data[0:spliter]
  test_iris_data = iris_data[spliter+1:len(iris_data)]
  return train_iris_data, test_iris_data


# input_sizeは行数
# hidden_sizeは列数
def generate_weight(input_size, hidden_size):
  return np.random.randn(input_size, hidden_size)
      
def generate_bias(hidden_size):
  return np.zeros(hidden_size)
      
def sigmoid(x):
  return 1 / (1 + np.exp())
      
def soft_max(a):
  c = np.max(a)
  exp_a = np.exp(a - c) #オーバーフロー対策
  sum_exp_a = np.sum(exp_a)
  return exp_a / sum_exp_a
  
  
epochs = 100  
iris_data = get_iris_data('./data.csv')
W1 = generate_weight(3, 100)
W2 = generate_weight(100, 100)
W3 = generate_weight(100, 3)
B1 = generate_bias(100)
B2 = generate_bias(100)
B3 = generate_bias(100)

train_iris_data, test_iris_data = split_iris_data(iris_data, 0.8)



