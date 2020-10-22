#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np
import tensorflow as tf
import nibabel as nib
import glob
import time
from tensorflow.keras.utils import to_categorical
from sys import stdout
import matplotlib.pyplot as plt
import matplotlib.image as mpim
from scipy.ndimage.interpolation import affine_transform
from augmentation import *
from sklearn.model_selection import train_test_split


physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True) 


Nclasses = 4
classes = np.arange(Nclasses)

# images lists ducati
# t1_list = sorted(glob.glob('/flush2/common/BRATS_2020/Training/*/*t1.nii'))
# t2_list = sorted(glob.glob('/flush2/common/BRATS_2020/Training/*/*t2.nii'))
# t1ce_list = sorted(glob.glob('/flush2/common/BRATS_2020/Training/*/*t1ce.nii'))
# flair_list = sorted(glob.glob('/flush2/common/BRATS_2020/Training/*/*flair.nii'))
# seg_list = sorted(glob.glob('/flush2/common/BRATS_2020/Training/*/*seg.nii'))

# # images lists harley
t1_list = sorted(glob.glob('/flush2/common/BRATS_2020/MICCAI_BraTS2020_TrainingData/*/*t1.nii'))
t2_list = sorted(glob.glob('/flush2/common/BRATS_2020/MICCAI_BraTS2020_TrainingData/*/*t2.nii'))
t1ce_list = sorted(glob.glob('/flush2/common/BRATS_2020/MICCAI_BraTS2020_TrainingData/*/*t1ce.nii'))
flair_list = sorted(glob.glob('/flush2/common/BRATS_2020/MICCAI_BraTS2020_TrainingData/*/*flair.nii'))
seg_list = sorted(glob.glob('/flush2/common/BRATS_2020/MICCAI_BraTS2020_TrainingData/*/*seg.nii'))

Nim = len(t1_list)

def load_img(img_files):
    ''' Load one image and its target form file
    '''
    N = len(img_files)
    # target
    y = nib.load(img_files[N-1]).get_fdata(dtype='float32', caching='unchanged')
    y = y[40:200,34:226,8:136]
    y[y==4]=3
      
    X_norm = np.empty((240, 240, 155, 4))
    for channel in range(N-1):
        X = nib.load(img_files[channel]).get_fdata(dtype='float32', caching='unchanged')
        brain = X[X!=0] 
        brain_norm = np.zeros_like(X) # background at -100
        norm = (brain - np.mean(brain))/np.std(brain)
        brain_norm[X!=0] = norm
        X_norm[:,:,:,channel] = brain_norm        
        
    X_norm = X_norm[40:200,34:226,8:136,:]    
    del(X, brain, brain_norm)
    
    return X_norm, y


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=4, dim=(160,192,128), n_channels=4, n_classes=4, shuffle=True, augmentation=False, patch_size=64, n_patches=8):
        'Initialization'
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data     
        X, y = self.__data_generation(list_IDs_temp)
        if self.augmentation == True:
            X, y = self.__data_augmentation(X, y)
        
        if index == self.__len__()-1:
            self.on_epoch_end()
        
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
  
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim))

        # Generate data
        for i, IDs in enumerate(list_IDs_temp):
            # Store sample
            X[i], y[i] = load_img(IDs)
            
        if self.augmentation == True:
            return X.astype('float32'), y
        else:
            return X.astype('float32'), to_categorical(y, self.n_classes)

    def __data_augmentation(self, X, y):
        'Apply augmentation'
        X_aug, y_aug = patch_extraction(X, y, sizePatches=self.patch_size, Npatches=self.n_patches)
        X_aug, y_aug = aug_batch(X_aug, y_aug, decisions)
                
        return X_aug, to_categorical(y_aug, self.n_classes)


# In[ ]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, Conv3DTranspose, Dropout, ReLU, LeakyReLU, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

class unet():
    def __init__(self, img_shape, seg_shape, class_weights, Nfilter_start=64, depth=4, batch_size=4):
        self.img_shape = img_shape
        self.seg_shape = seg_shape
        self.class_weights = class_weights
        self.Nfilter_start = Nfilter_start
        self.depth = depth
        self.batch_size = batch_size
        
        def dice(y_true, y_pred, w=self.class_weights):
            y_true = tf.convert_to_tensor(y_true, 'float32')
            y_pred = tf.convert_to_tensor(y_pred, y_true.dtype)

            num = tf.math.reduce_sum(tf.math.multiply(w, tf.math.reduce_sum(tf.math.multiply(y_true, y_pred), axis=[0,1,2,3])))
            den = tf.math.reduce_sum(tf.math.multiply(w, tf.math.reduce_sum(tf.math.add(y_true, y_pred), axis=[0,1,2,3])))+1e-5

            return 2*num/den
        
        def diceLoss(y_true, y_pred, w=self.class_weights):
            dice_score = dice(y_true, y_pred, w)
            
            return 1-dice_score
        
    
        inputs = Input(self.img_shape, name='input_image')     

        def encoder_step(layer, Nf, norm=True):
            x = Conv3D(Nf, kernel_size=4, strides=1, kernel_initializer='he_normal', padding='same')(layer)
            if norm:
                x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            x = Conv3D(Nf, kernel_size=4, strides=2, kernel_initializer='he_normal', padding='same')(x)
            if norm:
                x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            
            return x
        
        def bottlenek(layer, Nf):
            x = Conv3D(Nf, kernel_size=4, strides=2, kernel_initializer='he_normal', padding='same')(layer)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            for i in range(2):
                y = Conv3D(Nf, kernel_size=4, strides=1, kernel_initializer='he_normal', padding='same')(x)
                x = BatchNormalization()(y)
                x = Dropout(0.2)(x)
                x = LeakyReLU()(x)
                x = Concatenate()([x, y])
                
            return x

        def decoder_step(layer, layer_to_concatenate, Nf):
            x = Conv3DTranspose(Nf, kernel_size=4, strides=2, padding='same', kernel_initializer='he_normal')(layer)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Concatenate()([x, layer_to_concatenate])
            x = Conv3D(Nf, kernel_size=4, strides=1, padding='same', kernel_initializer='he_normal')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            return x

        layers_to_concatenate = []
        x = inputs

        # encoder
        for d in range(self.depth-1):
            if d==0:
                x = encoder_step(x, self.Nfilter_start*np.power(2,d), False)
            else:
                x = encoder_step(x, self.Nfilter_start*np.power(2,d))
            layers_to_concatenate.append(x)
        
        # bottlenek
        x = bottlenek(x, self.Nfilter_start*np.power(2,self.depth-1))

        # decoder
        for d in range(self.depth-2, -1, -1): 
            x = decoder_step(x, layers_to_concatenate.pop(), self.Nfilter_start*np.power(2,d))

        # classifier
        last = Conv3DTranspose(self.seg_shape[-1], kernel_size=4, strides=2, padding='same', kernel_initializer='he_normal', activation='softmax', name='output_generator')(x)

        # Create model
        self.model = Model(inputs=inputs, outputs=last)
        self.model.compile(loss=diceLoss, optimizer=Adam(lr=1e-4), metrics=['accuracy',dice])
  
            
    def train(self, train_gen, valid_gen, nEpochs, model_name):
        
        print('Training process:')
        callbacks = [ModelCheckpoint(model_name, verbose=1, save_best_only=True, save_weights_only=True), EarlyStopping(monitor='val_loss', patience=50)]
        history = self.model.fit(train_gen, validation_data=valid_gen, epochs=nEpochs, batch_size=self.batch_size, callbacks=callbacks)
        
        return history
        
    


# In[ ]:


imShape = (128, 128, 128, 4)
gtShape = imShape
class_weights = np.load('class_weights.npy')



# In[ ]:


from sklearn.model_selection import KFold

kf = KFold(n_splits=3, shuffle=True)
cv = 0
path = './RESULTS/'
if os.path.exists(path)==False:
    os.mkdir(path)
            
for idxTrain, idxValid in kf.split(np.arange(Nim)):
    cv +=1
    
    print('Cross Validation #{}'.format(cv))
    sets = {'train': [], 'valid': []}

    for i in idxTrain:
        sets['train'].append([t1_list[i], t2_list[i], t1ce_list[i], flair_list[i], seg_list[i]])
    for i in idxValid:
        sets['valid'].append([t1_list[i], t2_list[i], t1ce_list[i], flair_list[i], seg_list[i]])
    
    train_gen = DataGenerator(sets['train'], augmentation=True, patch_size=128, n_patches=1)
    valid_gen = DataGenerator(sets['valid'], augmentation=True, patch_size=128, n_patches=1)
    
    # train
    net = unet(imShape, gtShape, class_weights)
    h = net.train(train_gen, valid_gen, nEpochs=200, model_name= path+'model_cv{}.h5'.format(cv))
