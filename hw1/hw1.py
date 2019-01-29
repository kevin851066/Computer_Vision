
# coding: utf-8

# In[2]:


import numpy as np
from numpy import linalg as LA
import math
import cv2
import time
import sys

# useful function
def gaussian(x, sigma):
    return np.exp( (-(x) ** 2) / (2 * (sigma) ** 2) )

def rgb2gray(rgb, W, i):
    return np.dot(rgb[...,:3], W[i])

def Gs(sigma_s):
    r = 3 * sigma_s
    gs = []
    for i in range(1, 2*r+2):
        for j in range(1, 2*r+2):
            gs.append([i, j])
    gs = np.array(gs)
    center = gs[ (2 * r + 1) * r + r]
    buf = []
    for i in range(gs.shape[0]):
        buf.append(center-gs[i])
    gs = gaussian(LA.norm(buf, axis = 1), sigma_s) 
    gs = (np.array(gs)).reshape(2*r+1,2*r+1)
    return gs

def Gr_single(img_array, sigma_r, sigma_s):
    r = 3 * sigma_s
    center = img_array[r][r]
    img_array = img_array.reshape(4 * (r ** 2) + 4 * r + 1, 1)
    buf = []
    for i in range(img_array.shape[0]):
        buf.append(center - img_array[i])
    img_array = gaussian(LA.norm(buf, axis = 1), sigma_r)    
    img_array = ( np.array( img_array ) ).reshape( 2 * r + 1, 2 * r + 1 )
    return img_array

def Gr_color(img_array, sigma_r, sigma_s):
    r = 3 * sigma_s
    center = img_array[r][r]
    img_array = img_array.reshape(4 * (r ** 2) + 4 * r + 1, 1, 3)
    buf = []
    for i in range(img_array.shape[0]):
        buf.append(center - img_array[i])
    img_array = gaussian(LA.norm(buf, axis = 2), sigma_r)    
    img_array = ( np.array( img_array ) ).reshape( 2 * r + 1, 2 * r + 1 )
    return img_array

def padding_1D(img,sigma_s):  
    r = 3*sigma_s
    pad = np.pad(img, ( (r, r), (r, r) ), 'constant')
    return pad

def padding_3D(img,sigma_s):    
    r = 3*sigma_s
    pad = np.pad(img, ( (r, r), (r, r), (0, 0) ),'constant')
    return pad
        
def joint_bilateral_filter(source_image, guide_image, sigma_s, sigma_r): # guide_image is 2D
    r = 3 * sigma_s                                                     # source_image is 3D
    window_size = 2 * r + 1
    guide_image = np.array(guide_image)
    source_image = np.array(source_image)
    h = guide_image.shape[0] # height of source_img
    w = guide_image.shape[1] # width of source_img
    padding_guide = padding_1D(guide_image, sigma_s) # padding
    padding_source = padding_3D(source_image, sigma_s) 
    gs_array = np.array([])
    gs_array = Gs(sigma_s)  # calculate the gs matrix and remember it
    gr_array = np.array([])
    afterJBF = [] 
    for i in range(h):  # x
        for j in range(w): # y
            window_guide = padding_guide[i : i + window_size, j : j + window_size]
            window_source = padding_source[i : i + window_size, j : j + window_size, :] # Iq
            gr_array = Gr_single(window_guide, sigma_r, sigma_s)  # calculate the gr matrix and remember it
            denominator_array = gs_array * gr_array
            denominator = denominator_array.sum()
            stack_denominator = np.stack( (denominator_array, ) * 3, axis = 2)  #  use this to multiply with Iq
            numerator_array = stack_denominator * window_source
            numerator = numerator_array.sum(axis = 0)
            numerator = numerator.sum(axis = 0)
            Ip = numerator/denominator
            afterJBF.append(Ip)
    afterJBF = np.array(afterJBF)
    afterJBF = afterJBF.reshape(h, w, 3)
    return afterJBF

def bilateral_filter(source_image, guide_image, sigma_s, sigma_r): # guide_image is 3D 
    r = 3 * sigma_s                                                     # source_image is 3D
    window_size = 2 * r + 1
    guide_image = np.array(guide_image)
    source_image = np.array(source_image)
    h = guide_image.shape[0] # height of source_img
    w = guide_image.shape[1] # width of source_img
    padding_guide = padding_3D(guide_image, sigma_s) # padding
    padding_source = padding_3D(source_image, sigma_s) 
    gs_array = np.array([])
    gs_array = Gs(sigma_s)  # calculate the gs matrix and remember it
    gr_array = np.array([])
    afterJBF = [] 
    for i in range(h):  # x
        for j in range(w): # y
            window_guide = padding_guide[i : i + window_size, j : j + window_size, :]
            window_source = padding_source[i : i + window_size, j : j + window_size, :] # Iq
            gr_array = Gr_color(window_guide, sigma_r, sigma_s)  # calculate the gr matrix and remember it
            denominator_array = gs_array * gr_array
            denominator = denominator_array.sum()
            stack_denominator = np.stack( (denominator_array, ) * 3, axis = 2)  #  use this to multiply with Iq
            numerator_array = stack_denominator * window_source
            numerator = numerator_array.sum(axis = 0)
            numerator = numerator.sum(axis = 0)
            Ip = numerator/denominator
            afterJBF.append(Ip)
    afterJBF = np.array(afterJBF)
    afterJBF = afterJBF.reshape(h, w, 3)
    return afterJBF

# main program
# 66 candidates
Wr = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
Wg = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
Wb = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
Wrgb = []
sigma_s = [1, 2, 3]
sigma_r = [0.05, 0.1, 0.2]

for r in Wr:
    for g in Wg:
        for b in Wb:
            if abs( r + g + b - 1.0 ) < 10 ** ( -9 )  :
                Wrgb.append((r, g, b))
                
Wrgb = np.array(Wrgb)            

# read the image
if __name__ == '__main__':
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1],cv2.IMREAD_ANYCOLOR)

img = img/255
vote_board = np.zeros(66)
W = Wrgb
W = 10 * W
W = W.astype(int)

for s in sigma_s:
    for r in sigma_r:   # choose one (s,r) and have a contest (total 9 contests)
        cost_board = []
        source_guide = np.array([])
        source_guide = bilateral_filter(img, img, s, r) 
        for i in range(66):  # make 66 condidates
            gray = rgb2gray(img,Wrgb,i) # gray is a numpy array (198,200) 
            guide = joint_bilateral_filter(img, gray, s, r)
            cost = LA.norm(source_guide - guide)
            cost_board.append(cost)
        cost_board = np.array(cost_board) 
        x_index = -1
        for i in W:
            lose = 0
            game = 0
            x_index += 1
            y_index = -1
            for j in W:
                y_index += 1
                if (i[0] - j[0] == 1) and (j[1] - i[1] == 1):
                    game += 1
                    if cost_board[x_index] >= cost_board[y_index]:
                        break
                    else:
                        lose += 1
                elif (j[0] - i[0] == 1) and (i[1] - j[1] == 1):
                    game += 1
                    if cost_board[x_index] >= cost_board[y_index]:
                        break
                    else:
                        lose += 1
                elif (i[1] - j[1] == 1) and (j[2] - i[2] == 1):
                    game += 1
                    if cost_board[x_index] >= cost_board[y_index]:
                        break
                    else:
                        lose += 1
                elif (j[1] - i[1] == 1) and (i[2] - j[2] == 1):
                    game += 1
                    if cost_board[x_index] >= cost_board[y_index]:
                        break
                    else:
                        lose += 1
                elif (i[0] - j[0] == 1) and (j[2] - i[2] == 1):
                    game += 1
                    if cost_board[x_index] >= cost_board[y_index]:
                        break
                    else:
                        lose += 1
                elif (j[0] - i[0] == 1) and (i[2] - j[2] == 1):
                    game += 1
                    if cost_board[x_index] >= cost_board[y_index]:
                        break
                    else:
                        lose += 1
            if game == lose :
                vote_board[x_index] += 1
                
first_three_index = vote_board.argsort()[-3:][::-1]
first = rgb2gray(img, Wrgb, first_three_index[0])
second = rgb2gray(img, Wrgb, first_three_index[0])
third = rgb2gray(img, Wrgb, first_three_index[0])

cv2.imwrite('first_place_' + sys.argv[1], first * 255)
cv2.imwrite('second_place_' + sys.argv[1], second * 255)
cv2.imwrite('third_place_' + sys.argv[1], third * 255)

            


# In[ ]:




