import numpy as np
import matplotlib.pyplot as plt
import random
#pseudo inverse
'''
class PseudoInverseFNN:
    def __init__(self, inputSize, hidenNodeCount, outputSize):
        self._inputSize = inputSize
        self._hidenNodeCount = hidenNodeCount
        self._outputSize = outputSize

    def activationFunciton(self, m):
        r, l = m.shape
        res = []
        for i in range(r):
            t = []
            for j in range(l):
                t.append(1.0 / 1 + np.exp(-1.0*m[i][j]))
            res.append(t)
        return np.array(res)

    def makeRandP(self, n):
        condiction = min(n, self._hidenNodeCount)
        while True:
            P = np.random.rand(n, self._hidenNodeCount)
            rank = np.linalg.matrix_rank(P)
            if rank == condiction:
                break
        self._P = P

    def check(self,A,B):
        r,l = A.shape
        for i in range(r):
            print(A[i] - B[i])
        exit()

    def fit(self, x, y):
        x_c, x_r = x.shape
        assert x_r == self._inputSize
        y_c, y_r = y.shape
        assert y_r == self._outputSize
        self.makeRandP(x_c)
        pix = np.linalg.pinv(x)
        self._W1 = np.matmul(pix, self._P)

        #PP = np.matmul(x,self._W1)
        #self.check(PP, self._P)

        H = self.activationFunciton(self._P)
        piH = np.linalg.pinv(H)
        self._W2 = np.matmul(piH, y)

    def predict(self, x):
        r, l = x.shape
        assert l == self._inputSize
        t = np.matmul(x, self._W1)
        t = self.activationFunciton(t)
        t = np.matmul(t, self._W2)
        return t
'''
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
            print("正在训练第{}层".format(index+1))
            inyl = np.linalg.pinv(yl)
            err = self.check(np.matmul(yl, inyl))
            if index > 0 and err < self._error:
                print("精度满足，退出")
                break
            self._WV.append(inyl)
            yl = self.activationFunciton(np.matmul(yl,inyl))
            index += 1
        print("层数{}".format(len(self._WV) + 1))
        self._WL = np.matmul(np.linalg.pinv(yl), y)

    def predict(self, x):
        r, l = x.shape
        assert l == self._inputSize
        t = x
        for w in self._WV:
            t = np.matmul(t,w)
            t = self.activationFunciton(t)
        return np.matmul(t, self._WL)

if __name__ == "__main__":
    p = PseudoInverseFNN(1,1)
    x = []
    y = []
    for i in range(1000):
        a = i * 0.1
        x.append([a])
        #y.append([np.sin(a)])
        y.append([np.sin(a*2)+0.6])
    p.fit(np.array(x),np.array(y))
    yy = p.predict(np.array(x))
    plt.plot(x,y)
    plt.plot(x,yy)
    plt.show()
    #for i in range(len(x)):
    #    print(x[i], y[i], yy[i])
    '''
    for i in range(100):
        for j in range(100):
            a = i * 0.02
            b = j * 0.02
            x.append([a,b])
            y.append([np.sin(a) + np.sin(b)])
    '''

