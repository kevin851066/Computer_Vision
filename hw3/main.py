
# coding: utf-8

# In[1]:


import numpy as np
import cv2


# u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
# this function should return a 3-by-3 homography matrix
def solve_homography(u, v):
    N = u.shape[0]
    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')
	# if you take solution 2:
    A = np.zeros((2*N, 9))
    b = np.zeros((2*N, 1))
    # TODO: compute H from A and b
    for i in range(4):
        A[2 * i][0] = u[i][0]
        A[2 * i][1] = u[i][1]
        A[2 * i][2] = 1
        A[2 * i][6] = -( u[i][0] * v[i][0] )
        A[2 * i][7] = -( u[i][1] * v[i][0] )
        A[2 * i][8] = -v[i][0]
        A[2 * i + 1][3] = u[i][0]
        A[2 * i + 1][4] = u[i][1]
        A[2 * i + 1][5] = 1
        A[2 * i + 1][6] = -( u[i][0] * v[i][1] )
        A[2 * i + 1][7] = -( u[i][1] * v[i][1] )
        A[2 * i + 1][8] = -v[i][1]
    U, s, vh = np.linalg.svd(A.transpose() @ A, full_matrices = False)
    H = U[:, 8].reshape(3,3)
    return H
    # 求A^(-1)可能會造成結果不穩定 (A^T*A)^(-1)會比較好

# corners are 4-by-2 arrays, representing the four image corner (x, y) pairs
def transform(img, canvas, corners):
    h, w, ch = img.shape
    # TODO: some magic
    u = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
    H = solve_homography(u, corners)
    for x in range(w):
        for y in range(h):
            u_coor = np.array([x, y ,1])
            v_coor = np.dot(H, u_coor.reshape(3,1))
            v_coor = v_coor * (1 / v_coor[2][0])
            v_coor = v_coor.astype(int)
            canvas[v_coor[1][0]][v_coor[0][0]] = img[y][x]

def main():
    # Part 1    
    canvas = cv2.imread('./input/nyse.jpg')
    img1 = cv2.imread('./input/all.jpg')
    img2 = cv2.imread('./input/green.jpg')
    img3 = cv2.imread('./input/purple.jpg')
    img4 = cv2.imread('./input/yellow.jpg')
    img5 = cv2.imread('./input/red.jpg')

    corners1 = np.array([[69, 54], [244, 152], [29, 205], [219, 266]])
    corners2 = np.array([[77, 366], [192, 366], [67, 442], [185, 446]])
    corners3 = np.array([[725, 36], [846, 36], [725, 108], [846, 108]])
    corners4 = np.array([[679, 198], [757, 198], [676, 305], [760, 305]])
    corners5 = np.array([[818, 291], [930, 298], [819, 363], [933, 369]])

    # TODO: some magic
    transform(img1, canvas, corners1)
    transform(img2, canvas, corners2)
    transform(img3, canvas, corners3)
    transform(img4, canvas, corners4)
    transform(img5, canvas, corners5)
    cv2.imwrite('part1.png', canvas)

#  Part 2
    img6 = cv2.imread('./input/screen.jpg')
    # TODO: some magic
    corners6 = np.array([[1039, 369], [1102, 394], [982, 554], [1036, 599]])
    SideLengthQR = 300
    QR = np.zeros((300, 300, 3))
    # TODO: some magic
    u = np.array([[0, 0], [SideLengthQR, 0], [0, SideLengthQR], [SideLengthQR, SideLengthQR]])
    H = solve_homography(u, corners6)
    for x in range(SideLengthQR):
        for y in range(SideLengthQR):
            u_coor = np.array([x, y ,1])
            v_coor = np.dot(H, u_coor.reshape(3,1))
            v_coor = v_coor * (1 / v_coor[2][0])
            v_coor = v_coor.astype(int)
            QR[y][x] = img6[v_coor[1][0]][v_coor[0][0]]
    cv2.imwrite('part2.png', QR)

    # Part 3
    img_front = cv2.imread('./input/crosswalk_front.jpg')
    # TODO: some magic
    corners7 = np.array([[130, 141], [599, 143], [2, 280], [722, 278]])
    TopViewHeight = 130
    TopViewWidth = 450
    TopView = np.zeros((130, 450, 3))
    # TODO: some magic
    u = np.array([[0, 0], [TopViewWidth, 0], [0, TopViewHeight], [TopViewWidth, TopViewHeight]])
    H = solve_homography(u, corners7)
    for x in range(TopViewWidth):
        for y in range(TopViewHeight):
            u_coor = np.array([x, y ,1])
            v_coor = np.dot(H, u_coor.reshape(3,1))
            v_coor = v_coor * (1 / v_coor[2][0])
            v_coor = v_coor.astype(int)
            TopView[y][x] = img_front[v_coor[1][0]][v_coor[0][0]]

    cv2.imwrite('part3.png', TopView)


if __name__ == '__main__':
    main()

