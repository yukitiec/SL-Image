import pandas as pd
import cv2
from sklearn.utils import shuffle
import numpy as np

class Dataset():
  def __init__(self,scale_x,scale_y,img_size,src_train,src_test):
    self.src_train = src_train
    self.src_test = src_test
    self.scale_x = scale_x
    self.scale_y = scale_y
    self.img_size = img_size

    data_train,label_train,data_valid,label_valid,data_test,label_test = self.load_data()

    img_train, label_train = self.url2img(data_train,label_train)
    img_valid, label_valid = self.url2img(data_valid,label_valid)
    img_test, label_test = self.url2img(data_test,label_test)

    #when calculating (dividing process) the type have to be float
    image_train = img_train.astype("float32")
    image_valid = img_valid.astype("float32")
    image_test = img_test.astype("float32")

    #when training, it is convenient if the values are normalized
    #-1 ~ 1
    img_train = img_train/255
    img_train = 2*img_train - np.ones(img_train.shape)
    img_valid = img_valid/255
    img_valid = 2*img_valid - np.ones(img_valid.shape)
    img_test = img_test/255
    img_test = 2*img_test - np.ones(img_test.shape)

    # converting data to float32, especially float32
    self.img_train =np.expand_dims(np.asarray(img_train).astype(np.float32),1)
    self.label_train = np.asarray(label_train).astype(np.int32)
    self.img_valid = np.expand_dims(np.asarray(img_valid).astype(np.float32),1)
    self.label_valid = np.asarray(label_valid).astype(np.int32)
    self.img_test = np.expand_dims(np.asarray(img_test).astype(np.float32),1)
    self.label_test = np.asarray(label_test).astype(np.int32)

  def load_data(self):
    file_train = pd.read_csv(filepath_or_buffer = self.src_train)
    value_train = file_train.values
    file_test = pd.read_csv(filepath_or_buffer = self.src_test)
    value_test = file_test.values

    print(value_test[:5])
    ## ////////// テスト用とトレーニング用でデータをシャッフルする //////////////
    #for 2D shuffle
    value_train = shuffle(value_train,random_state=42) #random_state = integer ; fix randomness with shuffling
    value_test = shuffle(value_test,random_state=42)

    num_train = int(value_train.shape[0]*0.95)
    #DATA,LABEL
    #train,val
    data_train = value_train[:num_train,1:]
    label_train =  value_train[:num_train,0]
    data_valid = value_train[num_train:,1:]
    label_valid =  value_train[num_train:,0] #value_train.shape[0]//2
    #test
    data_test = value_test[:,1:]
    label_test = value_test[:,0]

    #check the distribution
    count_test = 0
    count_train = 0
    count_valid = 0
    for i in range(len(label_test)):
      if label_test[i] == 1:
        count_test += 1

    for i in range(len(label_train)):
      if label_train[i] == 1:
        count_train += 1
    
    for i in range(len(label_valid)):
      if label_train[i] == 1:
        count_valid += 1

    print('spatter label : : train : {}/{}, valid : {}/{} test : {}/{}'.format(count_train,len(label_train),count_valid,len(label_valid),count_test,len(label_test)))
    return data_train,label_train,data_valid,label_valid,data_test,label_test


  def url2img(self,data,label):
    image_data = np.zeros((len(data),int(self.img_size*self.scale_x),int(self.img_size*self.scale_y)))
    label_revised = []
    size = (int(self.img_size*self.scale_x),int(self.img_size*self.scale_y))
    for i,[name] in enumerate(data):
      if i % 100 == 99:
        print((len(data),i))
      #print(name)
      img=cv2.imread(name)
      if np.any(img) == None:
        continue
      elif img[0,0,0] != None :
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray,size,interpolation = cv2.INTER_AREA)
        image_data[i] = gray
        label_revised.append(label[i])
    print("length of dataset is ",len(label_revised))
    print("image data shape : ",image_data.shape)
    imagedata = image_data[:len(label_revised)]
    #imagedata = np.expand_dims(imagedata,1) #add color channel 1
    return imagedata,label_revised