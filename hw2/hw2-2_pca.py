
# coding: utf-8

# In[5]:


"""
• python3 hw2-2_pca.py $1 $2 $3
• $1: path of whole dataset
• $2: path of the input testing image
• $3: path of the output testing image reconstruct by all eigenfaces
• E.g., python3 hw2-2_pca.py hw2-2_data hw2-2_data/1_1.png hw2-2_output/output_pca.png
"""

import os
import sys
import cv2
import time 
import numpy as np

def zeroMean(dataMat):      
    meanVal = np.mean(dataMat, axis = 0)     #按列求均值，即求各个特征的均值
    newData = dataMat - meanVal
    return newData, meanVal

def pca(dataMat,n):
    newData, meanVal = zeroMean(dataMat)
    covMat = np.cov(newData, rowvar = 0) # Estimate a covariance matrix
    eigVals, eigVects = np.linalg.eig(np.mat(covMat)) # eigVals: column vector   # argsort: returns the indices that would sort an array
    eigValIndice = np.argsort(eigVals)                # eigVects: every row represents an eigen vect
    n_eigValIndice = eigValIndice[-1 : - (n + 1) : -1] #從最後一個開始，往前數n個
    n_eigVect = eigVects[:, n_eigValIndice]
    lowDDataMat = newData * n_eigVect
    reconMat = (lowDDataMat * n_eigVect.T) + meanVal
    return lowDDataMat, reconMat, n_eigVect, meanVal
    
train = []

for i in range(40):
    for j in range(10):
        if j < 7:
            file_name = str(i+1) + '_' + str(j+1) + '.png'
            img = cv2.imread(os.path.join(sys.argv[1], file_name), 0)
            img = img.flatten()
            train.append(img)
        else:
            break

train = np.array(train)

def reconstruct(img, n):
    lowDDataMat, reconMat, eigVec, meanface = pca(train, n)
    meanface = meanface[np.newaxis]
    p5 = eigVec.T * (img - meanface).T
    x = eigVec * p5 + meanface.T
    x = np.real(x)
    x = x.reshape(56, 46)
    img = np.real(img)
    img = img.reshape(56, 46)
    return x

img = cv2.imread(sys.argv[2], 0)
img = img.flatten()
rec = reconstruct(img, 279)
cv2.imwrite(sys.argv[3], rec)
print('Saving image success!')
    
# print('train shape: {} // test shape: {}'.format(train.shape, test.shape))


# In[94]:


# meanface

# newData, meanface = zeroMean(train)
# meanface = meanface.reshape(56, 46, 1)
# cv2.imwrite('meanface/meanface.jpg', meanface)

# reconstruct image

# def mse(imageA, imageB):
#     s = np.array(imageA.astype("float") - imageB.astype("float"))
#     err = np.sum(s * s)
#     err /= float(imageA.shape[0] * imageA.shape[1]) 
#     return err

# def reconstruct(img, n):
#     lowDDataMat, reconMat, eigVec, meanface = pca(train, n)
#     meanface = meanface[np.newaxis]
#     p5 = eigVec.T * (img - meanface).T
#     x = eigVec * p5 + meanface.T
#     x = np.real(x)
#     x = x.reshape(56, 46)
#     img = np.real(img)
#     img = img.reshape(56, 46)
#     err = mse(img, x)
#     print('mse_{}'.format(n), err)
#     return x

# for i in [5, 50, 150, 279]:
#     rec = reconstruct(img, i)
#     cv2.imwrite('rec_img/rec_{}.jpg'.format(i), rec)


# In[167]:


# img = cv2.imread("C:/Users/user/Desktop/junior/computer vision/hw2/hw2-2_data/8_6.png", 0)
# img = img.flatten()
# img = img[np.newaxis]
# print(img.shape)


# In[7]:


# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt

# color = []

# for i in range(40):
#     for j in range(3):
#         if j == 3:
#             break
#         else:
#             color.append(i+1)    
            
# color = np.array(color)

# lowDDataMat, reconMat, n_eigVect, meanVal = pca(test, 100)
# # print(lowDDataMat.shape)
# x_tsne = TSNE(n_components = 2).fit_transform(lowDDataMat)
# x_min, x_max = np.min(x_tsne, 0), np.max(x_tsne, 0)
# x_tsne = (x_tsne - x_min) / (x_max - x_min)
# fig = plt.figure()
# plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c = color, cmap = plt.cm.spectral)
# plt.title('PCA Scattering Dim = 100')
# plt.show()

