import cv2 as cv
import numpy as np
import os
import random
from tqdm import tqdm
import tensorflow as tf
import WeiNi

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

def RandH():
    H = [[1,0,0],[0,1,0],[0,0,1]]
    H_l = []

    H[0][0] = H[0][0] + random.random()/50 - 0.01
    H[0][1] = H[0][1] + random.random()/50 - 0.01
    H[0][2] = H[0][2] + random.random()*400 - 200
    H[1][0] = H[1][0] + random.random()/50 - 0.01
    H[1][1] = H[1][1] + random.random()/50 - 0.01
    H[1][2] = H[1][2] + random.random()*400 - 200
    H[2][0] = H[2][0] + random.random()/2000 - 0.00025
    H[2][1] = H[2][1] + random.random()/2000 - 0.00025
    for i in range(3):
        for j in range(3):
            H_l.append(H[i][j])
    H_l=H_l[:8]
    H = np.matrix(H)
    return H, H_l

def MakeHFromHL(HL):
    H = []
    tmp = HL[0:3]
    H.append(tmp)
    tmp = HL[3:6]
    H.append(tmp)
    tmp = HL[6:]
    tmp.append(1.0)
    H.append(tmp)
    return H


def ConstructTringDataSet(img):
    h, w = img.shape[:2]
    print(h,w)
    # 随机生成透视变换矩阵
    features = []
    for i in range(20):
        H,HL = RandH()
        imgH = cv.warpPerspective(img,H,dsize=(w,h))
        detector = cv.xfeatures2d.SIFT_create()
        keypoints_sift, descriptors = detector.detectAndCompute(imgH, None)
        for j in range(len(descriptors)):
            features.append((descriptors[j],HL))
    random.shuffle(features)
    X = []
    Y = []
    for i in range(len(features)):
        X.append(features[i][0])
        Y.append(features[i][1])
    return X,Y

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

    # 训练FNN
    tX, tY = ConstructTringDataSet(img_l)
    # 定义网络结构
    # 输入层
    x = tf.placeholder("float",[None,128],name='x')
    # 标签
    y = tf.placeholder("float",[None,8],name='y')
    l_1 = tf.layers.dense(inputs=x,units=100,activation=tf.nn.relu,name='layer_1')
    l_2 = tf.layers.dense(inputs=l_1, units=100, activation=tf.nn.relu, name='layer_2')
    l_3 = tf.layers.dense(inputs=l_2, units=100, activation=tf.nn.relu, name='layer_3')
    l_o = tf.layers.dense(inputs=l_3, units=1,name='out')
    # 定义损失函数
    loss = tf.square(l_o - y)
    # 优化方法 梯度下降
    #train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    train_step = tf.train.AdadeltaOptimizer(learning_rate=0.05, rho=0.95).minimize(loss)
    #return train_step
    tX = np.array(tX)
    tY = np.array(tY)
    init_op = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init_op)
    batch = 1000
    size = len(tX)
    print(size, batch)
    for i in range(0):
        index = (i + batch) % size
        end = index + batch
        tmp_x = np.array(tX[index:end])
        tmp_y = np.array(tY[index:end])
        _,l = sess.run((train_step,loss), feed_dict={x:tmp_x,y:tmp_y})
        if i % 1000 == 0:
            sum = 0
            for j in range(len(l[0])):
                sum += l[0][j]
            print("迭代第{}次:{}".format(i,sum))

    # 测试

    # 当筛选后的匹配对大于4时， 计算视角变化矩阵
    if len(matches) > 4:
        # 获取匹配对的点坐标
        ptsA = np.float32([kp_l[i].pt for (_, i) in matches])
        ptsB = np.float32([kp_r[i].pt for (i, _) in matches])

        # 计算视角变化矩阵
        (H, status) = cv.findHomography(ptsA, ptsB, cv.RANSAC, 1)
        print(H)
        result = warpTwoImages(img_r, img_l, H)
        print(H)
        cv.imshow('SIFT', result)
        cv.waitKey()
        print("done")
    else:
        print("拼接失败")

def test():
    img = cv.imread("./img/l_2.jpg", cv.IMREAD_GRAYSCALE)
    h,w = img.shape
    img = cv.resize(img,(int(w/8),int(h/8)))
    ConstructTringDataSet(img)
    exit(0)
if __name__ == "__main__":
    #test()
    print("\t\t\t【图像拼接】")
    TestSIFT()