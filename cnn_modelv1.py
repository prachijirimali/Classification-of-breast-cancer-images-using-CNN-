import pandas as pd
import numpy as np
import tensorflow as tf
import os
import cv2
import imageio
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
import shutil
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from numpy.random import seed
seed(101)
tf.random.set_seed(101)
img0 = os.listdir('C://Prachi/dataset/0')
df_data0 = pd.DataFrame(img0, columns=['image_id'])
#print(df_data0.head())

##Function to extract target
def extract_target(x):
    a = x.split('_')
    # target = index 4
    b = a[4]
    #print(b)
    # the ytarget i.e. 1 or 2 is the 5th index of the string --> class1
    target = b[5]
    return target

x = df_data0
df_data0['target'] = df_data0['image_id'].apply(extract_target)
#print(df_data0.head())

img1 = os.listdir('C://Prachi/dataset/1')
df_data1 = pd.DataFrame(img1, columns=['image_id'])
#print(df_data1.head())

x = df_data1
df_data1['target'] = df_data1['image_id'].apply(extract_target)
#print(df_data1.head())

#print(df_data0['target'].value_counts())
#print(df_data1['target'].value_counts())
df_data = pd.concat([df_data0, df_data1], axis=0).\
    reset_index(drop=True)
#print(df_data['target'].value_counts())
y = df_data['target']

#TRAIN-TEST SPLIT
df_train, df_test = train_test_split(df_data, test_size=0.3, random_state=101, stratify=y)
#print(df_train['target'].value_counts())
#print(df_test['target'].value_counts())
'''
base_dir = 'base_dir'
os.mkdir(base_dir)

# create a path to 'base_dir' to which we will join the names of the new folders
# train_dir
train_dir = os.path.join(base_dir, 'train_dir')
os.mkdir(train_dir)

# test_dir
test_dir = os.path.join(base_dir, 'test_dir')
os.mkdir(test_dir)

# create new folders inside train_dir
no_idc = os.path.join(train_dir, 'no_idc')
os.mkdir(no_idc)
idc = os.path.join(train_dir, 'idc')
os.mkdir(idc)

# create new folders inside test_dir
no_idc = os.path.join(test_dir, 'no_idc')
os.mkdir(no_idc)
idc = os.path.join(test_dir, 'idc')
os.mkdir(idc)

#print(os.listdir('base_dir/train_dir'))
#print(os.listdir('base_dir/test_dir'))

df_data.set_index('image_id', inplace=True)
train_list = list(df_train['image_id'])

#image_list = os.listdir('all_images_dir')
all_images_dir = 'all_images_dir'
#os.mkdir(all_images_dir)

# TRANSFER TRAINING AND TESTING IMAGES

for image in train_list:

    # the id in the csv file does not have the .tif extension therefore we add it here
    fname = image
    # get the label for a certain image
    target = df_data.loc[image, 'target']

    # these must match the folder names
    if target == '0':
        label = 'no_idc'
    if target == '1':
        label = 'idc'

    # source path to image
    src = os.path.join(all_images_dir, fname)
    # destination path to image
    dst = os.path.join(train_dir, label, fname)
    # move the image from the source to the destination
    shutil.move(src, dst)
    train_path = 'training_images'
    valid_path = 'testing_images'

# check how many val images we have in each folder

# Transfer the val images
test_list = list(df_test['image_id'])
for image in test_list:

    # the id in the csv file does not have the .tif extension therefore we add it here
    fname = image
    # get the label for a certain image
    target = df_data.loc[image, 'target']

    # these must match the folder names
    if target == '0':
        label = 'no_idc'
    if target == '1':
        label = 'idc'

    # source path to image
    src = os.path.join(all_images_dir, fname)
    # destination path to image
    dst = os.path.join(test_dir, label, fname)
    # move the image from the source to the destination
    shutil.move(src, dst)
print(len(os.listdir('base_dir/train_dir/no_idc')))
print(len(os.listdir('base_dir/test_dir/no_idc')))

train_path = 'base_dir/train_dir'
valid_path = 'base_dir/test_dir'


num_train_samples = len(df_data)
num_val_samples = len(df_test)
train_batch_size = 10
val_batch_size = 10


train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)
IMAGE_SIZE = 50
datagen = ImageDataGenerator(rescale=1.0/255)

train_gen = datagen.flow_from_directory(train_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=train_batch_size,
                                        class_mode='categorical')

val_gen = datagen.flow_from_directory(valid_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=val_batch_size,
                                        class_mode='categorical')

# Note: shuffle=False causes the test dataset to not be shuffled
test_gen = datagen.flow_from_directory(valid_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=1,
                                        class_mode='categorical',
                                        shuffle=False)
kernel_size = (3,3)
pool_size= (2,2)
first_filters = 32
second_filters = 64
third_filters = 128

dropout_conv = 0.3
dropout_dense = 0.3


model = Sequential()
model.add(Conv2D(first_filters, kernel_size, activation = 'relu',
                 input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(dropout_dense))
model.add(Dense(2, activation = "softmax"))

model.summary()

model.compile(Adam(lr=0.0001), loss='binary_crossentropy',
              metrics=['accuracy'])
filepath = "model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,
                             save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3,
                              verbose=1, mode='max', min_lr=0.00001)

callbacks_list = [checkpoint, reduce_lr]

history = model.fit_generator(train_gen, steps_per_epoch=train_steps,
                              validation_data=val_gen,
                              validation_steps=val_steps,
                              epochs=60, verbose=1,
                              callbacks=callbacks_list)
'''
# get the metric names so we can use evaulate_generator
model.metrics_names
# Here the best epoch will be used.

model.load_weights('model.h5')

val_loss, val_acc = \
model.evaluate_generator(test_gen,
                        steps=len(df_test))

print('val_loss:', val_loss)
print('val_acc:', val_acc)

# make a prediction
predictions = model.predict_generator(test_gen, steps=len(df_test), verbose=1)
print(predictions.shape)
# This is how to check what index keras has internally assigned to each class.
print(test_gen.class_indices)
# Put the predictions into a dataframe.
# The columns need to be ordered to match the output of the previous cell

df_preds = pd.DataFrame(predictions, columns=['idc', 'no_idc'])

print(df_preds.head())

# Get the true labels
y_true = test_gen.classes

# Get the predicted labels as probabilities
y_pred = df_preds['no_idc']
from sklearn.metrics import roc_auc_score

print(roc_auc_score(y_true, y_pred))
