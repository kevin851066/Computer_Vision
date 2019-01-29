
# coding: utf-8

# In[12]:


"""
• python3 hw2-2_lda.py $1 $2
• $1: path of whole dataset
• $2: path of the first 1 Fisherface
• E.g., python3 hw2-2_lda.py hw2-2_data hw2-2_output/output_fisher.png
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
    return lowDDataMat, n_eigVect

def lda(lowDDataMat, classes, n_data_each_class): # N筆資料，C個類別，先用PCA把資料降至N-C維，得到lowDDataMat
    lowDDataMat = np.real(lowDDataMat)            # lowDDataMat: 經eigenvector映射的資料們(降維的向量們) shape: N * (N - C) 
    lowdim = lowDDataMat.shape[1]
    
    mus = [] # 每個class的mu
    for i in range(classes):
        sl = lowDDataMat[n_data_each_class * i : n_data_each_class * i + n_data_each_class, :]
        mu = np.mean(sl, axis = 0)
        mus.append(mu)

    mus = np.array(mus)
    mus = mus.reshape(classes,lowdim)
    
    total_mu = np.zeros((1, lowdim)) #計算整個資料的mu
    for i in range(classes):
        total_mu += n_data_each_class * mus[i, :]
    total_mu /= float(n_data_each_class * classes)

    sw = np.zeros( (lowdim, lowdim) )
    for i in range(classes):
        swi = np.zeros( (lowdim, lowdim) )
        for j in range(n_data_each_class):
            x = lowDDataMat[i * n_data_each_class + j, :] 
            swi += np.dot( (x - mus[i]).T, x - mus[i] ) 
        sw += swi

    sb = np.zeros((lowdim, lowdim))
    for i in range(classes):
        sb += n_data_each_class * np.dot( (mus[i, :] - total_mu).T, mus[i, :] - total_mu )

    lda_eigVal, lda_eigVec = np.linalg.eig( np.dot( np.linalg.inv(sw), sb) )
    lda_eigValIndice = np.argsort(lda_eigVal)                
    lda_n_eigValIndice = lda_eigValIndice[-1 : - classes : -1] #從最後一個開始，往前數 C-1個，C = 類別數
    lda_n_eigVec = lda_eigVec[:, lda_n_eigValIndice]
    return lda_n_eigVec  # 用lda找最適合區分資料的基(前C-1個)
    
train = []
train_label = []

for i in range(40):
    for j in range(10):
        file_name = str(i+1) + '_' + str(j+1) + '.png'
        img = cv2.imread(os.path.join(sys.argv[1], file_name), 0)
        # print(img.shape)
        img = img.flatten()
        train.append(img)
        train_label.append(i+1)

train = np.array(train)
train_label = np.array(train_label)

# print('train shape: {} // test shape: {}'.format(train.shape, test.shape))
# print('train label shape: {} // test label shape: {}'.formast(train_label.shape, test_label.shape))

lowDDataMat, n_eigVect = pca(train, 360)

# find fisherface
lda_proj = lda(lowDDataMat, 40, 10)
fisherfaces = np.dot(n_eigVect, lda_proj)
fisherfaces = np.real(fisherfaces)
# choose fist 30 fisherface (dim:30)
# d_30 = fisherfaces[:, 0 : 30]
# new_test, test_mean = zeroMean(test)
# test_proj = np.dot(d_30.T, new_test.T)

fisherface = fisherfaces[:, 1]
fisherface = fisherface - np.min(fisherface)
fisherface = fisherface * ( 255 / np.max(fisherface) )
fisherface = fisherface.reshape(56,46)
cv2.imwrite(sys.argv[2], fisherface)
print('Saving image success!')
# print(lowDDataMat_test.shape)
# p = n_eigVect.T * test.T


# In[2]:


# print(n_eigVect.shape)


# In[3]:


# print(lowDDataMat.shape)


# In[14]:


# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt





# plot lda scatter
# color = []

# for i in range(40):
#     for j in range(3):
#         if j == 3:
#             break
#         else:
#             color.append(i+1)    
            
# color = np.array(color)

# x_tsne = TSNE(n_components = 2).fit_transform(test_proj.T)
# fig = plt.figure()
# plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c = color, cmap = plt.cm.spectral)
# plt.title('LDA Scattering Dim = 30')
# plt.show()



# plot fisherface 

# for i in range(5):
#     fisherface = fisherfaces[:, i]
#     fisherface = fisherface - np.min(fisherface)
#     fisherface = fisherface * ( 255 / np.max(fisherface) )
#     fisherface = fisherface.reshape(56,46)
#     cv2.imwrite('fisherface/fisherface_{}.jpg'.format(i+1), fisherface)


# In[16]:


# train PCA 
# n = 3, 10, 39
# lowDDataMat_3, eigVect_3 = pca(train, 3)
# lowDDataMat_10, eigVect_10 = pca(train, 10) 
# lowDDataMat_39, eigVect_39 = pca(train, 39) 
# print('eigVect_3: ', eigVect_3.shape)
# print('eigVect_10: ', eigVect_10.shape)
# print('eigVect_39: ', eigVect_39.shape)


# In[17]:


# train LDA 
# n = 3, 10, 39
# lowDDataMat_forLDA, eigVect_forLDA = pca(train, 280)
# lda_p = lda(lowDDataMat_forLDA, 40, 7)
# ffaces = np.dot(eigVect_forLDA, lda_p)
# ffaces = np.real(ffaces)
# d_3 = ffaces[:, 0 : 3]
# d_10 = ffaces[:, 0 : 10]
# d_39 = ffaces[:, 0 : 39]
# print('d_3: ', d_3.shape)
# print('d_10: ', d_10.shape)
# print('d_39: ', d_39.shape)

# new_test, test_mean = zeroMean(test)
# test_proj = np.dot(d_30.T, new_test.T)


# In[82]:


# three split
# def sp(data):
#     split_1 = []
#     split_2 = []
#     split_3 = []
#     split_label_1 = []
#     split_label_2 = []
#     split_label_3 = []

#     for i in range(40):
#         for j in range(7):
#             if j < 2:
#                 split_1.append(data[i * 7 + j, :])
#                 split_label_1.append(i+1)
#             elif j >= 2 and j < 4:
#                 split_2.append(data[i * 7 + j, :])
#                 split_label_2.append(i+1)
#             else:
#                 split_3.append(data[i * 7 + j, :])
#                 split_label_3.append(i+1)

#     split_1 = np.array(split_1).reshape((80, 39))
#     split_2 = np.array(split_2).reshape((80, 39))
#     split_3 = np.array(split_3).reshape((120, 39))
    
#     split_label_1 = np.array(split_label_1)
#     split_label_2 = np.array(split_label_2)
#     split_label_3 = np.array(split_label_3)
#     return split_1, split_2, split_3, split_label_1, split_label_2, split_label_3

# new, mean = zeroMean(train)
# new_test, mean_test = zeroMean(test)
# p = eigVect_39.T * (new).T
# l,e = pca(test, 39)
# t = e.T * (new_test).T
# t = t.T
# t = np.real(t)
# p = p.T
# p = np.real(p)
# sp1, sp2, sp3, spl1, spl2, spl3 = sp(p)
# print(sp1.shape)
# print(sp2.shape)
# print(sp3.shape)
# l = np.hstack((spl1,spl2))
# print(l.shape)
# print(p.shape)
# print(t.shape)


# In[84]:



# from sklearn.neighbors import KNeighborsClassifier

# def acc(label_1, label_2):
#     score = 0
#     for i in range(len(label_1)):
#         if label_1[i] == label_2[i]:
#             score += 1
#     score = score / float(len(label_1))
#     return score

# def KNN(k, train_data, train_label, valid_data, valid_label):
# #     d = ffaces[:, 0 : n]
#     knn = KNeighborsClassifier(n_neighbors = k)
#     knn.fit(train_data,train_label)
#     predict_label = knn.predict(valid_data)
# #     print(predict_label)
#     return acc(predict_label, valid_label)

# t1 = np.vstack((sp1,sp2))
# l1 = np.hstack((spl1,spl2))
# t2 = np.vstack((sp2,sp3))
# l2 = np.hstack((spl2,spl3))
# t3 = np.vstack((sp1,sp3))
# l3 = np.hstack((spl1,spl3))
# lt = np.hstack((spl1,spl2))
# print(lt.shape)
# lt = np.hstack((lt,spl3))

# print(lt.shape)
# print('n = 39, k = 1', KNN(1, p, lt, t, test_label) )
# print('group 1 as valid, k = 1', KNN(1, t2, l2, sp1, spl1) )
# print('group 2 as valid, k = 1', KNN(1, t3, l3, sp2, spl2) )
# print('mean of k = 1', ( KNN(1, t1, l1, sp3, spl3) + KNN(1, t2, l2, sp1, spl1) + KNN(1, t3, l3, sp2, spl2) ) / float(3) )

# print('group 3 as valid, k = 3', KNN(3, t1, l1, sp3, spl3) )
# print('group 1 as valid, k = 3', KNN(3, t2, l2, sp1, spl1) )
# print('group 2 as valid, k = 3', KNN(3, t3, l3, sp2, spl2) )
# print('mean of k = 3', ( KNN(3, t1, l1, sp3, spl3) + KNN(3, t2, l2, sp1, spl1) + KNN(3, t3, l3, sp2, spl2) ) / float(3) )

# print('group 3 as valid, k = 5', KNN(5, t1, l1, sp3, spl3) )
# print('group 1 as valid, k = 5', KNN(5, t2, l2, sp1, spl1) )
# print('group 2 as valid, k = 5', KNN(5, t3, l3, sp2, spl2) )
# print('mean of k = 5', ( KNN(5, t1, l1, sp3, spl3) + KNN(5, t2, l2, sp1, spl1) + KNN(5, t3, l3, sp2, spl2) ) / float(3) )


# In[ ]:


# get_ipython().system('jupyter nbconvert --to script hw2-3.ipynb')

