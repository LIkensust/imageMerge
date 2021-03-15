import cv2 as cv
import numpy as np
import os
import random
#import tensorflow as tf
from numpy import float32

# 伪逆训练器
class PseudoInverseFNN:
    def __init__(self, inputSize, outputSize, e=0.01, maxLayer=10):
        self._inputSize = inputSize
        self._hidenNodeCount = inputSize
        self._outputSize = outputSize
        self._error = e
        self._maxLayer = maxLayer

    def activationFunciton(self, m):
        r, l = m.shape
        res = []
        for i in range(r):
            t = []
            for j in range(l):
                t.append(1.0 / (1 + np.exp(-1.0*m[i][j])))
            res.append(t)
        return np.array(res)

    def check(self,A):
        r,l = A.shape
        sum = 0
        for i in range(r):
            for j in range(l):
                if i == j:
                    sum += (A[i][j] - 1)**2
                else:
                    sum += A[i][j]**2
        return sum

    def fit(self, x, y):
        x_c, x_r = x.shape
        assert x_r == self._inputSize
        y_c, y_r = y.shape
        assert y_r == self._outputSize

        self._WV = []
        index = 0
        yl = x
        while index < self._maxLayer:
            print("正在训练第{}层...".format(index+1))
            print("计算伪逆矩阵...")
            inyl = np.linalg.pinv(yl)
            print("验证精度...")
            err = self.check(np.matmul(yl, inyl))
            print("误差[{}],要求精度[{}]".format(err, self._error))
            if index > 0 and err < self._error:
                print("精度满足，退出")
                break
            self._WV.append(inyl.copy())
            yl = self.activationFunciton(np.matmul(yl,inyl))
            index += 1
        if index >= self._maxLayer:
            print("不收敛")
            #exit()
        print("层数{}".format(len(self._WV) + 1))
        self._WL = np.matmul(np.linalg.pinv(yl), y)


    def predict(self, x):
        r, l = x.shape
        assert l == self._inputSize
        t = x

        for w in self._WV:
            print(t.shape)
            print(w.shape)
            t = np.matmul(t,w)
            t = self.activationFunciton(t)
        return np.matmul(t, self._WL)

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

# 随机生成透视变换矩阵以及对应的列向量
def RandH():
    H = [[1,0,0],[0,1,0],[0,0,1]]
    H_l = []
    H[0][0] = random.uniform(1.8,2.3)
    H[0][1] = random.uniform(0,0.3)
    H[0][2] = random.uniform(-550,0)
    H[1][0] = random.uniform(0.6,1.2)
    H[1][1] = random.uniform(1.7,2.5)
    H[1][2] = random.uniform(-160,-140)
    H[2][0] = random.uniform(0.001,0.004)
    H[2][1] = random.uniform(-0.0005,-0.0004)

    for i in range(3):
        for j in range(3):
            H_l.append(H[i][j])
    H_l=H_l[:8]
    H = np.matrix(H)
    return H, H_l

def MakeFeature(img):
    detector = cv.xfeatures2d.SIFT_create()
    keypoints_sift, descriptors = detector.detectAndCompute(img, None)
    return np.array(descriptors)


def split_list(l, n=64, new=[]):
    if len(l) <= n:
        new.append(l)
        return new
    else:
        new.append(l[:n])
        return split_list(l[n:], n)

def ConstructTringDataSet(img):
    h, w = img.shape[:2]
    print(h,w)
    # 随机生成透视变换矩阵
    features = []
    for i in range(250):
        print(i)
        H,HL = RandH()
        imgH = cv.warpPerspective(img,H,dsize=(w,h))
        detector = cv.xfeatures2d.SIFT_create()
        # 关键点与描述符
        _, descriptors = detector.detectAndCompute(imgH, None)
        # 4个特征点为一组 组合特征
        #for j in range(len(descriptors)):
        #    features.append((descriptors[j],HL))
        if descriptors is None:
            continue
        dd = split_list(descriptors, 4)
        j = 0
        for d in dd:
            if len(d) < 4:
                break
            nd = np.array(d[0].tolist() + d[1].tolist() + d[2].tolist() + d[3].tolist())
            features.append((nd, HL))
            j += 1
            if(j>2):
                break
    random.shuffle(features)
    X = []
    Y = []
    for i in range(len(features)):
        X.append(features[i][0])
        Y.append(features[i][1])
    return X,Y

def MakeHFromHL(HL):
    HL = HL.tolist()
    H = []
    tmp = HL[0:3]
    H.append(tmp)
    tmp = HL[3:6]
    H.append(tmp)
    tmp = HL[6:]
    tmp.append(1.0)
    H.append(tmp)
    return np.array(H)

def openImage(path):
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    h,w = img.shape
    img = cv.resize(img,(int(w/10),int(h/10)))
    return img

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

def main():
    # 打开图像
    img_l = openImage("./img/left.jpg")
    img_r = openImage("./img/right.jpg")
    x, y = ConstructTringDataSet(img_l)
    x = np.array(x[:200])
    y = np.array(y[:200])
    print(x.shape)
    print(y.shape)

    # 训练FNN
    fnn = PseudoInverseFNN(128*4,8,0.01)
    print('start traing')
    fnn.fit(x,y)

    h, w = img_l.shape[:2]

    # 对第二张图 抽取特征描述符
    features = MakeFeature(img_r)
    dd = split_list(features, 4)
    px = []
    for d in dd:
        if len(d) < 4:
            break
        nd = np.array(d[0].tolist() + d[1].tolist() + d[2].tolist() + d[3].tolist())
        px.append(nd)
    px = np.array(px)
    HS = fnn.predict(px)

    for i in range(len(HS)):
        H = MakeHFromHL(HS[i])
        print(H)
        result = warpTwoImages(img_r, img_l, H)
        cv.imshow('SIFT', result)
        cv.waitKey()
    '''
    # 可视化测试
    h, w = img_l.shape[:2]
    H = MakeHFromHL(y[0])
    print(H)
    imgH = cv.warpPerspective(img_l, H, dsize=(w, h))
    cv.imshow('SIFT', imgH)
    cv.waitKey()

    py = fnn.predict(x)
    H = MakeHFromHL(py[0])
    print(H)
    imgH = cv.warpPerspective(img_l, H, dsize=(w, h))
    cv.imshow('SIFT', imgH)
    cv.waitKey()
    '''

if __name__ == "__main__":
    main()
