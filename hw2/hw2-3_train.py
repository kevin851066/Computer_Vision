
# coding: utf-8

# In[9]:


"""
• python3 hw2-3_train.py $1
• $1: directory of the hw2-3_data folder
• E.g., python3 hw2-3_train.py ./hw2/hw2-3_data/
"""
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:

import cv2
import os
import sys
import time 
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, Activation, MaxPooling2D
from keras.models import Model

from keras.optimizers import Adam
# from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical

x_train = []
y_train = []
x_valid = []
y_valid = []

num_classes = 10

starttime = time.clock() 

for i in range(10):
    for j in range(6000):
        if j < 5000:
            j = "%04d" % j
            file_name = 'train/' + 'class_' + str(i) + '/' + str(j) + '.png'
            image = cv2.imread(os.path.join(sys.argv[1], file_name), 0)
            x_train.append(image)
            y_train.append(to_categorical(i, num_classes))
        else:
            j = "%04d" % j
            file_name = 'valid/' + 'class_' + str(i) + '/' + str(j) + '.png'
            image = cv2.imread(os.path.join(sys.argv[1], file_name), 0)
            x_valid.append(image)
            y_valid.append(to_categorical(i, num_classes))
            
x_train = np.array(x_train)
y_train = np.array(y_train)
x_valid = np.array(x_valid)
y_valid = np.array(y_valid)

print('x_train shape: {} // y_train shape: {}'.format(x_train.shape, y_train.shape))
print('x_valid shape: {} // y_valid shape: {}'.format(x_valid.shape, y_valid.shape))

endtime = time.clock() 
print (endtime - starttime) 

x_train = np.expand_dims(x_train, axis = -1) / 255
x_valid = np.expand_dims(x_valid, axis = -1) / 255


# In[35]:


def Lenet_5():
    img_input = Input( shape = (28, 28, 1) )
    co1 = Conv2D(6, (5, 5), padding = 'valid', name = 'co1')(img_input)
    co1 = Activation('tanh')(co1)
    mp1 = MaxPooling2D(pool_size = 2, strides = 2, padding = 'SAME')(co1)
    co2 = Conv2D(16, (5, 5), padding = 'valid', name = 'co2')(mp1)
    co2 = Activation('tanh')(co2)
    mp2 = MaxPooling2D(pool_size = 2, strides = 2, padding = 'SAME', name = 'mp2')(co2)
    flat = Flatten()(mp2)
    fc1 = Dense(120, activation = 'tanh', name = 'fc1')(flat)
    fc2 = Dense(84, activation = 'tanh', name = 'fc2')(fc1)
    fc3 = Dense(10, activation = 'softmax', name = 'fc3_sm')(fc2)
    
    model = Model(img_input, fc3)
    return model

model = Lenet_5()
model.summary()
model.compile(loss = 'categorical_crossentropy', 
              optimizer = Adam(lr = 10 ** ( -4 )), 
              metrics = ['accuracy'])


# In[12]:


epochs = 10
batch_size = 32

history = model.fit(x_train,
                    y_train,
                   batch_size = batch_size,
                   epochs = epochs,
                   validation_data = (x_valid, y_valid),
                   verbose = 1)


# In[14]:


# # plot learning curve
# l = history.history['loss']
# vl = history.history['val_loss']
# acc = history.history['acc']
# vacc = history.history['val_acc']

# plt.figure(figsize=(12, 5))
# plt.subplot(121)
# plt.plot(np.arange(epochs)+1, l, 'b', label='train loss')
# plt.plot(np.arange(epochs)+1, vl, 'r', label='valid loss')
# plt.xlabel("epochs")
# plt.ylabel("loss")
# plt.title("loss curve")
# plt.legend(loc='best')

# plt.subplot(122)
# plt.plot(np.arange(epochs)+1, acc, 'b', label='train accuracy')
# plt.plot(np.arange(epochs)+1, vacc, 'r', label='valid accuracy')
# plt.xlabel("epochs")
# plt.ylabel("accuracy")
# plt.title("accuracy curve")
# plt.legend(loc='best')
# plt.tight_layout()

# plt.show()


# In[13]:


from keras.models import load_model
model.save('model/model.lenet_5_test')
# # 刪除既有模型變數
# del model 
# # 載入模型
# model = load_model('model/model.lenet_5')


# In[77]:


# from keras import backend as K
# from scipy.misc import imsave
# from keras.models import load_model

# # # 刪除既有模型變數
# # del model 
# # 載入模型
# model = load_model('model/model.lenet_5')

# layer_dict = dict([(layer.name, layer) for layer in model.layers])

# # input_img = cv2.imread("C:/Users/user/Desktop/junior/computer vision/hw2/hw2-3_data/train/class_0/0000.png", 0)

# layer_name = 'co2'
# filter_index = 5  # can be any integer from 0 to 511, as there are 512 filters in that layer

# # build a loss function that maximizes the activation
# # of the nth filter of the layer considered
# layer_output = layer_dict[layer_name].output
# print(layer_output)
# loss = K.mean(layer_output[:, :, :, filter_index])
# print(loss)
# # compute the gradient of the input picture wrt this loss
# input_img = model.input
# grads = K.gradients(loss, input_img)[0]
# print(grads)

# # normalization trick: we normalize the gradient
# grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5 )
# # this function returns the loss and grads given the input picture
# iterate = K.function([input_img], [loss, grads])

# # we start from a gray image with some noise
# input_img_data = np.random.random((1, 28, 28, 1)) * 125 + 128.
# # run gradient ascent for 20 steps
# step = 200
# for i in range(200):
#     loss_value, grads_value = iterate([input_img_data])
#     input_img_data += grads_value * step

# # util function to convert a tensor into a valid image
# def deprocess_image(x):
#     # normalize tensor: center on 0., ensure std is 0.1
#     x -= x.mean()
#     x /= (x.std() + 1e-5 )
#     x *= 0.1

#     # clip to [0, 1]
#     x += 0.5
#     x = np.clip(x, 0, 1)

#     # convert to RGB array
#     x *= 255
# #     x = x.transpose((1, 2, 0))
#     x = np.clip(x, 0, 255).astype('uint8')
#     return x

# img = input_img_data[0]
# img = deprocess_image(img)
# img = img - np.min(img)
# img = img * (255 / np.max(img) )
# # print(img)
# # print(img.shape)
# cv2.imwrite('filter/high_filter_6.png', img)


# In[16]:


# output = model.get_layer(name = 'co1').output
# print(output)


# In[38]:


# plot low, high level features

# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt

# visual_val = []
# for i in range(10):
#     for j in range(5000, 5100):
#         image = cv2.imread("C:/Users/user/Desktop/junior/computer vision/hw2/hw2-3_data/valid/class_{}/{}.png".format(i,j), 0)
#         visual_val.append(image)

# visual_val = np.array(visual_val)
# visual_val = np.expand_dims(visual_val, axis = -1) / 255

# features = model.predict(visual_val)
# model_extractfeatures = Model(input = model.input, output = model.get_layer('mp2').output)
# hl_features = model_extractfeatures.predict(visual_val)
# print(hl_features.shape)
# hl_features = hl_features.reshape(1000, 4*4*16)
# print(hl_features.shape)

# color = []

# for i in range(10):
#     for j in range(100):
#         if j == 100:
#             break
#         else:
#             color.append(i+1)    
            
# color = np.array(color)

# x_tsne = TSNE(n_components = 2).fit_transform(hl_features)
# fig = plt.figure()
# plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c = color, cmap = plt.cm.spectral)
# plt.title('High-Level Features')
# plt.show()

