import tensorflow as tf
import numpy as np
from keras.models import *
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, TensorBoard
from datetime import datetime
import sgydata
#np.random.seed(5)
np.random.seed(15)

class FCNloca(object):

  def __init__(self, img_rows = 70, img_cols = 4096):

    self.img_rows = img_rows
    self.img_cols = img_cols

  def load_data(self):
    wave_train,loca_train=sgydata.load_sgylist_xyz1(sgylist=['./waveform_data/','training_samples.txt'],
            sgyr=[0,-1,1],xr=[39,0.1,160],yr=[-135,0.1,150],zr=[-5,2,40],r=0.5,
            shiftdata=[list(range(20,50))+list(range(-200,-20)),1])
    loca_train = np.reshape(loca_train,(len(loca_train),160,150,40))
    print('end load_data()...')
    return wave_train, loca_train

  def get_network(self):

    inputs = Input((self.img_rows, self.img_cols,3))
    
    conv1 = Conv2D(32, kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(32, kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = MaxPooling2D(pool_size=(1, 2))(conv1)
    print("pool1 shape:",conv1.shape)

    conv1 = Conv2D(64, kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = Conv2D(64, kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(1, 2))(conv1)
    print("pool1 shape:",pool1.shape)

    conv2 = Conv2D(128, kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(1, 4))(conv2)
    print("pool2 shape:",pool2.shape)

    conv3 = Conv2D(256,kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256,kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 4))(conv3)
    print("pool3 shape:",pool3.shape)

    conv4 = Conv2D(512,kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, kernel_size=(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = Conv2D(512,kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = Conv2D(512,kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = Conv2D(512,kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.2)(conv4)
    pool4 = MaxPooling2D(pool_size=(4, 6))(drop4)
    print("pool4 shape:", pool4.shape)

    conv5 = Conv2D(1024,kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024,kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = Conv2D(1024,kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = Conv2D(1024,kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = Conv2D(2048,kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = Dropout(0.5)(conv5)
    conv5 = Conv2D(2048,kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = Conv2D(1024,kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = Conv2D(1024,kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = Conv2D(1024,kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = Conv2D(1024,kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,3))(drop5))
    print("Upsample1 shape:", up6.shape)
    conv6 = Conv2D(512,kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up6)
    conv6 = Conv2D(512,kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = Conv2D(512,kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = Dropout(0.2)(conv6)
    
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (5,5))(conv6))
    print("Upsample2 shape:",up7.shape)
    conv7 = Conv2D(256,kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up7)
    conv7 = Conv2D(256,kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    conv8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,1))(conv7))
    print("Upsample3 shape:",conv8.shape)
    conv8 = Conv2D(128,kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = Conv2D(128,kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    conv9 = Conv2D(64,kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv9 = Conv2D(64,kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    
    conv10 = Conv2D(40, 1, activation = 'sigmoid')(conv9)
    

    model = Model(input = inputs, output = conv10)
    print('conv10:', conv10.shape)

    model.compile(optimizer = Adam(lr = 5e-6), loss = 'binary_crossentropy', metrics = ['accuracy'])
#    model.compile(optimizer = 'sgd', loss = losses.mean_squared_error)
    return model

  def train(self):
    print("loading data")
    wave_train, loca_train = self.load_data()
    print("loading data done")
#    mirrored_strategy = tf.distribute.MirroredStrategy()
#    with mirrored_strategy.scope():
    model = self.get_network()
    print("got network")
    callbacks = [
  # Write TensorBoard logs to ./logs directory
    TensorBoard(log_dir='./log/{}'.format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S")), histogram_freq=1),
    ModelCheckpoint('FCNloca.hdf5', monitor='val_loss',verbose=1, save_best_only=True)]
    print('Fitting model...')
    hist=model.fit(wave_train, loca_train, batch_size=4, epochs=200, verbose=1, validation_split=0.1, shuffle=True, callbacks=callbacks)
    f=open('FCNloca.log','w')
    f.write(str(hist.history))
    f.close()


if __name__ == '__main__':
    fcnloca = FCNloca()
    fcnloca.train()




