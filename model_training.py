import tensorflow as tf
import keras.models
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from keras.layers import GlobalMaxPooling2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from imutils import paths
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizer_experimental import adam
import glob
import cv2
import argparse
import random
import os



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Create 4 virtual GPUs with 1GB memory each
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024),
         tf.config.LogicalDeviceConfiguration(memory_limit=1024),
         tf.config.LogicalDeviceConfiguration(memory_limit=1024),
         tf.config.LogicalDeviceConfiguration(memory_limit=1024),])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
# Load in the data

plt.close("all")

##-----------------------------------------
# import os

# path ="D:/Mine/Working/COVID_Image classification/data/coivd19_x_ray"
# #we shall store all the file names in this list
# filelist = []

# for root, dirs, files in os.walk(path):
# 	for file in files:
#         #append the file name to the list
# 		filelist.append(os.path.join(root,file))

# #print all the file names
# for name in filelist:
#     print(name)


myLabels = []

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

dirName = 'D:\Mine\Working\COVID_Image classification\data'
# Get the list of all files in directory tree at given path
listOfFiles = getListOfFiles(dirName)
images = []
labels = []
valid_images =[".jpg", ".gif", ".png", ".tga", ".jpeg"]
for elem in listOfFiles:
    ext = os.path.splitext(elem)[1]
    if ext.lower() not in valid_images:
        continue
    print(elem)        
    c = cv2.imread(elem)
    images.append(cv2.resize(c, [32, 32]))    
    labels.append(elem.split('\\')[-2])
    l = elem.split('\\')[-2]
    if l not in myLabels:
        myLabels.append(l)
    
images = np.array(images) / 255.0
#abels = np.array(labels)

lb = LabelEncoder()
labels = lb.fit_transform(labels)
#labels = to_categorical(labels)

(x_train, x_test, y_train, y_test) = train_test_split(images, labels,
                                                  test_size = 0.20, stratify = labels, random_state = 42)
trainAug = ImageDataGenerator(
	rotation_range=15,
	fill_mode="nearest")
##----------------------------------------
y_train = y_train.flatten()

print(x_train.shape, y_train.shape)
print(type(x_train), type(y_train))

# flatten the label values
#y_train= y_train.flatten()

# visualize data by plotting images
fig, ax = plt.subplots(5, 5)
k = 0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i])
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(y_train[i])
plt.show()


# number of classes
K = len(set(y_train))

# calculate total number of classes
# for output layer
print("number of classes:", K)

# Build the model using the functional API
# input layer



i = Input(shape=x_train[0].shape)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dropout(0.2)(x)

# Hidden layer
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)

# last hidden layer i.e.. output layer
x = Dense(K, activation='softmax')(x)

model = Model(i, x)

# model description
model.summary()



# Compile
model.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])
#INIT_LR = 1e-3
# opt = adam(lr=INIT_LR, decay=INIT_LR / 50)

#model.compile(loss="binary_crossentropy", optimizer='adam',
#	metrics=["accuracy"])

print(x_train.shape, y_train.shape)

# Fit
r = model.fit(
   x_train, y_train, epochs=50)

# H = model.fit_generator(
# 	trainAug.flow(x_train, y_train, batch_size=128),
# 	epochs=50)

#batch_size = 32
#data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
#    width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

#train_generator = data_generator.flow(x_train, y_train, batch_size)
#steps_per_epoch = x_train.shape[0] // batch_size

#r = model.fit(train_generator, validation_data=(x_test, y_test),
#              steps_per_epoch=steps_per_epoch, epochs=50)

# Plot accuracy per iteration
plt.plot(r.history['accuracy'], label='acc', color='red')
#plt.plot(r.history['val_accuracy'], label='val_acc', color='green')




# label mapping
plt.figure()
myLabels.sort()
# select the image from our test dataset
image_number = 22
for image_number in range(28):
    # display the image
    plt.imshow(x_test[image_number])
    
    # load the image in an array
    n = np.array(x_test[image_number])
    
    # reshape it
    p = n.reshape(1, 32, 32, 3)
    
    # pass in the network for prediction and
    # save the predicted label    
    predicted_label = myLabels[model.predict(p).argmax()]
    
    # load the original label
    original_label = myLabels[y_train[image_number]]
    
    
    # display the result
    print("Original label is {} and predicted label is {}".format(
        original_label, predicted_label))


# save the model
model.save('geeksforgeeks.h5')

'''
m = keras.models.load_model('geeksforgeeks.h5')
k=0
plt.figure(1,figsize=(10, 10))
for i in range(5):
    for j in range(5):
        while k < 25:
            r = random.randint(0, 1000)
            l = labels[y_test[r]]
            if l == 'leopard' or l == 'tiger' or l == 'lion' or l == 'wolf' or l == 'fox':
                #ax[i][j].imshow(x_test[r], aspect='auto')
                plt.subplot(5, 5, k + 1)
                plt.imshow(x_test[r])
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                original_label = labels[y_test[r]]
                n = np.array(x_test[r])
                p = n.reshape(1, 32, 32, 3)
                predicted_label = labels[m.predict(p).argmax()]
                plt.xlabel(predicted_label)
                print("Original label is {} and predicted label is {}".format(
                    original_label, predicted_label))
                k = k + 1
plt.savefig('saved_figure.png')
plt.show()
'''

'''
m = keras.models.load_model('geeksforgeeks.h5')
k = 0
plt.figure(figsize=(10, 10))

for i in range(25):
    r = random.randint(0, 1000)
    l = labels[y_test[r]]
    if l == 'leopard' or l == 'tiger' or l == 'lion' or l == 'wolf' or l == 'fox':
        plt.subplot(5,5, i+1)
        plt.imshow(x_test[r], aspect='auto')

        original_label = labels[y_test[r]]
        n = np.array(x_test[r])
        p = n.reshape(1, 32, 32, 3)
        predicted_label = labels[m.predict(p).argmax()]

        plt.xlabel(predicted_label)
        print("Original label is {} and predicted label is {}".format(
            original_label, predicted_label))
        i = i + 1
plt.show()
'''
