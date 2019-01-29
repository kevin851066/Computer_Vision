import cv2
import time
import math
import numpy as np
from numpy import linalg as LA
from skimage import color

# mean = 0.0
# standard_deviation = 4.0
# N = 4096
# window_size = 27
# padding_size = int( (window_size - 1) / 2 )

def preprocess(sample, scale): # make the data created by gaussian bound in a scale (here is 26)
    sample = sample - np.min(sample)
    sample = sample * ( scale / np.max(sample) )
    sample -= scale / 2
    sample = np.around(sample)
    return sample

def gaussian_2d(mean ,standard_deviation, N):
    sample = np.random.normal( loc = mean, scale = standard_deviation, size = 2 * N )   
    sample = preprocess(sample, 26)
    sample = sample.reshape(4096, 2)
    return sample

def padding_1D(img, padding_size):  
    pad = np.pad(img, ( (padding_size, padding_size), (padding_size, padding_size) ), 'constant')
    return pad

def padding_3D(img, padding_size):  # since window size is 27, so we pad 13 layers of zeros outside the image
    pad = np.pad(img, ( (padding_size, padding_size), (padding_size, padding_size), (0, 0) ), 'constant')
    return pad
        
def binary_function(p, q):
    b_arr = p - q
    b_arr = ( b_arr > 0 ).astype(int)
    return b_arr # binary array (4096, )
    
def bx(img_grayscle, p, q, disp):
    n = p.shape[0]
    s = 0 
    bx_arr = np.zeros(n).astype(int)
    p_x = ( p[:, 0] - disp ).astype(int) 
    p_y = ( p[:, 1] ).astype(int)
    q_x = ( q[:, 0] - disp ).astype(int) 
    q_y = ( q[:, 1] ).astype(int)
    bx_arr = binary_function( img_grayscle[p_y, p_x], img_grayscle[q_y, q_x] ) 
    return bx_arr #binary array (4096, )
    
def SAD(Ix, Iy):  # input are intensity
    sad = np.sum( np.absolute(Ix - Iy), axis = 1 )
    return sad

def weight_function(Ix, Ip, Iq):    
    w = np.maximum( SAD(Ix, Ip), SAD(Ix, Iq) )
    return w
    
def binary_mask(img, center_x, center_y, p, q):
    n = p.shape[0]
    p_x = ( p[:, 0] ).astype(int)
    p_y = ( p[:, 1] ).astype(int)
    q_x = ( q[:, 0] ).astype(int)
    q_y = ( q[:, 1] ).astype(int)
    Ix = img[center_y][center_x]
    w = weight_function( Ix, img[p_y, p_x], img[q_y, q_x] )
    weight_index_arr = np.argsort(w)
    T = w[ weight_index_arr[ int(n / 4) - 1 ] ]
    w = ( w <= T ).astype(int)
    return w # binary array

# def accumulated_weight(img_pad, center_x, center_y, occ_map, window_size, lumbda_c, lumbda_e, disp): # occ_map : the part of occlusion_map under the window
#     r = int( (window_size - 1) / 2 )
#     all_one_matrix = np.ones((window_size, window_size)).astype(int)
#     target_matrix = np.bitwise_and(all_one_matrix, occ_map)
#     target_coord = np.argwhere(target_matrix == 1)
#     target_coord -= np.array([r,r])
#     dist_matrix = Euclidean_distance(target_coord)
#     target_coord_color += np.array([center_x, center_y])
#     c_dist_matrix = color_distance( img_pad[center_y][center_x], img_pad[ target_coord_color[:, 1] ,target_coord_color[:, 0] ] ) 
#     accu_weight = np.sum( np.exp( -( dist_matrix / lumbda_e + c_dist_matrix / lumbda_c ) ), axis = 0 ) 
#     return dist_matrix

# def Euclidean_distance(target_coord):
#     dist_matrix = LA.norm(target_coord, axis = 1, keepdims = True)
#     return dist_matrix

# def color_distance(Cp, Cq):
#     c_dist_matrix = LA.norm(Cp - Cq, axis = 1, keepdims = True)
#     return c_dist_matrix

def computeDisp(Il, Ir, Il_g, Ir_g, max_disp):
    Il = Il.astype(int)
    Ir = Ir.astype(int)
    Il_g = Il_g.astype(int)
    Ir_g = Ir_g.astype(int)

    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype = np.uint8)
    labels_r = np.zeros((h, w), dtype = np.uint8)

    # >>> Cost computation
    tic = time.time()
    # TODO: Compute matching cost from Il and Ir
    Il_g_pad = padding_1D(Il_g, 13)
    Ir_g_pad = padding_1D(Ir_g, 13)
    Il_pad = padding_3D(Il, 13)
    Ir_pad = padding_3D(Ir, 13)
    Il_lab = color.rgb2lab(Il_pad / 255)
    Ir_lab = color.rgb2lab(Ir_pad / 255)
    
    sample_pair_coord_p = gaussian_2d(0.0, 4.0, 4096) # center at (0,0)
    sample_pair_coord_q = gaussian_2d(0.0, 4.0, 4096) # center at (0,0)

    for y in range(14 - 1, h - 1 + 14): # right disparity
        Bx_r_arr = np.zeros( (w, 4096) ).astype(int) # store Bx_r
        for x in range(14 - 1, w - 1 + 14): # (x,y) is the center of window
            base = np.array([x, y])
            modified_sample_pair_coord_p = base + sample_pair_coord_p
            modified_sample_pair_coord_q = base + sample_pair_coord_q
            B_mask = binary_mask(Il_lab, x, y, modified_sample_pair_coord_p, modified_sample_pair_coord_q) 
            Bx_l = bx(Il_g_pad, modified_sample_pair_coord_p, modified_sample_pair_coord_q, 0) 
            Bx_r = bx(Ir_g_pad, modified_sample_pair_coord_p, modified_sample_pair_coord_q, 0) 
            Bx_r_arr[x - 13] = Bx_r
            C = np.count_nonzero( np.bitwise_and( np.bitwise_xor( Bx_l, Bx_r ), B_mask ) )
            C_min = C
            xd = 0  # disparity (default is 0, since when the window is on the left-most side of img)
            if x == 13:
                xd = 0
            else:
                for d in range(1, max_disp):
                    if x - d - (14 - 1) < 0: # check if the window is out of image
                        break
                    else:
                        Bxd_r = Bx_r_arr[x - 13 - d]
                        C_new = np.count_nonzero( np.bitwise_and( np.bitwise_xor( Bx_l, Bxd_r ), B_mask ) )
                        if C_new <= C_min:
                            C_min = C_new
                            xd = d
            labels[y - 13][x - 13] = xd
    
    for y in range(14 - 1, h - 1 + 14): # left disparity
        Bx_l_arr = np.zeros( (w, 4096) ).astype(int) # store Bx_l
        for x in range(14 - 1, w - 1 + 14): # (x,y) is the center of window
            base = np.array([x, y])
            modified_sample_pair_coord_p = base + sample_pair_coord_p
            modified_sample_pair_coord_q = base + sample_pair_coord_q
            B_mask = binary_mask(Ir_lab, x, y, modified_sample_pair_coord_p, modified_sample_pair_coord_q) 
            Bx_l = bx(Il_g_pad, modified_sample_pair_coord_p, modified_sample_pair_coord_q, 0) 
            Bx_r = bx(Ir_g_pad, modified_sample_pair_coord_p, modified_sample_pair_coord_q, 0) 
            Bx_l_arr[x - 13] = Bx_l
            C = np.count_nonzero( np.bitwise_and( np.bitwise_xor( Bx_l, Bx_r ), B_mask ) )
            C_min = C
            xd = 0  # disparity (default is 0, since when the window is on the left-most side of img)
            if x == w - 1 + 13: # if x is the last column
                xd = 0
            else:
                for d in range(1, max_disp):
                    if x + d - (14 - 1) > w - 1: # check if the window is out of image
                        break
                    else:
                        if x == 13: # if x is the first column
                            Bxd_l = bx(Il_g_pad, modified_sample_pair_coord_p, modified_sample_pair_coord_q, - d)
                            Bx_l_arr[x - 13 + d] = Bxd_l
                            C_new = np.count_nonzero( np.bitwise_and( np.bitwise_xor( Bx_r, Bxd_l ), B_mask ) )
                            if C_new < C_min:
                                C_min = C_new
                                xd = d
                        else: # if x is not the first column
                            if d == max_disp - 1:
                                Bxd_l = bx(Il_g_pad, modified_sample_pair_coord_p, modified_sample_pair_coord_q, - d)
                                Bx_l_arr[x - 13 + d] = Bxd_l
                            else:
                                Bxd_l = Bx_l_arr[x - 13 + d] 
                            C_new = np.count_nonzero( np.bitwise_and( np.bitwise_xor( Bx_r, Bxd_l ), B_mask ) )
                            if C_new < C_min:
                                C_min = C_new
                                xd = d
            labels_r[y - 13][x - 13] = xd

    toc = time.time()
    print('* Elapsed time (cost computation): %f sec.' % (toc - tic))

    # >>> Cost aggregation
    tic = time.time()
    # TODO: Refine cost by aggregate nearby costs
    
    toc = time.time()
    print('* Elapsed time (cost aggregation): %f sec.' % (toc - tic))

    # >>> Disparity optimization
    tic = time.time()
    # TODO: Find optimal disparity based on estimated cost. Usually winner-take-all.
    toc = time.time()
    print('* Elapsed time (disparity optimization): %f sec.' % (toc - tic))

    # >>> Disparity refinement
    tic = time.time()
    # TODO: Do whatever to enhance the disparity map
    # ex: Left-right consistency check + hole filling + weighted median filtering
    occlusion_map = np.zeros((h, w))  
    for x in range(w):
        for y in range(h):
            if abs( labels[y][x] - labels_r[y][x - labels[y][x]] ) <= 1:
                occlusion_map[y][x] = 255
            else:
                occlusion_map[y][x] = 0
    occlusion_map = (occlusion_map > 0).astype(int) # valid pixels are 1, invalid are 0
    for y in range(h): 
        for x in range(w):
            if occlusion_map[y][x] == 0:
                left_disp_candidate = labels[y][x]
                right_disp_candidate = labels[y][x]
                for xx in range(w):
                    if x - xx < 0:
                        break
                    else: 
                        if occlusion_map[y][x - xx] == 1:
                            left_disp_candidate = labels[y][x - xx]
                            break
                for xx in range(w):
                    if x + xx > w - 1:
                        break
                    else:
                        if occlusion_map[y][x + xx] == 1:
                            right_disp_candidate = labels[y][x + xx]
                            break
                labels[y][x] = np.minimum(left_disp_candidate, right_disp_candidate)
    new = np.zeros((h,w))            
    new = cv2.medianBlur(labels,7)
                
    toc = time.time()
    print('* Elapsed time (disparity refinement): %f sec.' % (toc - tic))
    
    return new


def main():
    print('Tsukuba')
    img_left = cv2.imread('./testdata/tsukuba/im3.png')
    img_right = cv2.imread('./testdata/tsukuba/im4.png')
    img_left_grayscale = cv2.imread('./testdata/tsukuba/im3.png', 0)
    img_right_grayscale = cv2.imread('./testdata/tsukuba/im4.png', 0)
    max_disp = 15
    scale_factor = 16
    labels = computeDisp(img_left, img_right, img_left_grayscale, img_right_grayscale, max_disp)
    cv2.imwrite('tsukuba.png', np.uint8(labels * scale_factor))

    print('Venus')
    img_left = cv2.imread('./testdata/venus/im2.png')
    img_right = cv2.imread('./testdata/venus/im6.png')
    img_left_grayscale = cv2.imread('./testdata/venus/im2.png', 0)
    img_right_grayscale = cv2.imread('./testdata/venus/im6.png', 0)
    max_disp = 20
    scale_factor = 8
    labels = computeDisp(img_left, img_right, img_left_grayscale, img_right_grayscale, max_disp)
    cv2.imwrite('venus.png', np.uint8(labels * scale_factor))

    print('Teddy')
    img_left = cv2.imread('./testdata/teddy/im2.png')
    img_right = cv2.imread('./testdata/teddy/im6.png')
    img_left_grayscale = cv2.imread('./testdata/teddy/im2.png', 0)
    img_right_grayscale = cv2.imread('./testdata/teddy/im6.png', 0)
    max_disp = 60
    scale_factor = 4
    labels = computeDisp(img_left, img_right, img_left_grayscale, img_right_grayscale, max_disp)
    cv2.imwrite('teddy.png', np.uint8(labels * scale_factor))

    print('Cones')
    img_left = cv2.imread('./testdata/cones/im2.png')
    img_right = cv2.imread('./testdata/cones/im6.png')
    img_left_grayscale = cv2.imread('./testdata/cones/im2.png', 0)
    img_right_grayscale = cv2.imread('./testdata/cones/im6.png', 0)
    max_disp = 60
    scale_factor = 4
    labels = computeDisp(img_left, img_right, img_left_grayscale, img_right_grayscale, max_disp)
    cv2.imwrite('cones.png', np.uint8(labels * scale_factor))


if __name__ == '__main__':
    main()
