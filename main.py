import cv2 as cv
import numpy as np
import os

from numpy import float32
def TestLa():
    # 构造灰度图
    img = cv.imread("./img/left.jpg",cv.IMREAD_GRAYSCALE)
    (height, width) = img.shape
    img = cv.resize(img, (int(width/8),int(height/8)))
    (height, width) = img.shape
    cv.imshow('test',img)
    cv.waitKey()
    img_new = np.zeros([height-2,width-2])
    # 拉普拉斯卷积
    #laCore = np.array(([-1,-1,-1],[-1,8,-1],[-1,-1,-1]))
    laCore = np.array(([0.125,0.125,0.125],[0.125,0.125,0.125],[0.125,0.125,0.125]))

    img_new = cv.filter2D(img, -1, laCore)
    #for i in range(1,height-2):
    #    print(i,height)
    #    for j in range(1,width-2):
    #        tmp = img[i,j]*4 - img[i-1,j] - img[i+1,j] -img[i,j-1] - img[i, j+1]
    #        img_new[i-1,j-1]=tmp

    cv.imshow('test',img_new)
    cv.waitKey()

def get_img_keypoints_features(path):
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    h,w = img.shape
    img = cv.resize(img,(int(w/8),int(h/8)))
    detector = cv.xfeatures2d.SIFT_create()
    keypoints_sift, descriptors = detector.detectAndCompute(img, None)
    img_k = cv.drawKeypoints(img, keypoints_sift, img)
    #cv.imshow('SIFT', img_k)
    #cv.waitKey()
    return img,img_k,keypoints_sift,descriptors

def warpTwoImages(img1, img2, H):
    '''warp img2 to img1 with homograph H'''
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    print(pts1)
    pts2 = float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv.perspectiveTransform(pts2, H)

    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

    result = cv.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    return result

def TestSIFT():
    img_l, img_l_k, kp_l, f_l = get_img_keypoints_features("./img/l_2.jpg")
    img_r, img_r_k, kp_r, f_r = get_img_keypoints_features("./img/r_2.jpg")
    matcher = cv.DescriptorMatcher_create('BruteForce')
    rawMatches = matcher.knnMatch(f_l, f_r, 2)
    matches = []
    for m in rawMatches:
        # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
        if len(m) == 2 and m[0].distance < m[1].distance * 0.7:
            # 储存两个点在featuresA， featuresB中的索引值
            matches.append((m[0].trainIdx, m[0].queryIdx))

    # 当筛选后的匹配对大于4时， 计算视角变化矩阵
    if len(matches) > 4:
        # 获取匹配对的点坐标
        ptsA = np.float32([kp_l[i].pt for (_, i) in matches])
        ptsB = np.float32([kp_r[i].pt for (i, _) in matches])

        # 计算视角变化矩阵
        (H, status) = cv.findHomography(ptsA, ptsB, cv.RANSAC, 1)
        result = warpTwoImages(img_r, img_l, H)
        cv.imshow('SIFT', result)
        cv.waitKey()
    else:
        print("拼接失败")

if __name__ == "__main__":
    print("\t\t\t【图像拼接】")
    TestSIFT()