import numpy as np
import random
import matplotlib.pyplot as plt

class FNN:
    # isize 样本的维度
    # osize 标签的维度
    # nsize 隐藏层节点的数量
    def __init__(self, isize, osize, nsize):
        self._isize = isize
        self._osize = osize
        self._nsize = nsize

    # 随机产生矩阵P
    def makeP(self, M, L):
        P = []
        for i in range(M):
            tmp = []
            for j in range(L):
                tmp.append(random.uniform(0,1000))
            P.append(tmp)
        return np.array(P)

    # 激活函数（sigmoid）
    def acf(self, A):
        n,m = A.shape
        H = []
        for i in range(n):
            tmp = []
            for j in range(m):
                tmp.append(1.0 / (np.exp(-1.0*A[i][j]) + 1))
            H.append(tmp)
        return np.array(H)

    # 训练
    def fit(self,x,y):
        M, xnum = x.shape
        tM, ynum = y.shape
        assert M == tM
        assert xnum == self._isize
        assert ynum == self._osize

        # randomly generate M*L matrix P
        self.P = self.makeP(M, self._nsize)
        # 计算权重w1
        self.W1 = np.matmul(np.linalg.pinv(x), self.P)
        # 计算隐藏层输出
        H = self.acf(self.P)
        # 计算权重w2
        self.W2 = np.matmul(np.linalg.pinv(H), y)
        # 输出一下P的大小和秩，观察P是否是满秩的
        print('p的大小{}, 秩{}'.format(self.P.shape, np.linalg.matrix_rank(self.P)))

    # 预测
    def predict(self,x):
        # 第一次输出
        H = self.acf(np.matmul(x,self.W1))
        # 全连接层的输出
        return np.matmul(H,self.W2)

if __name__ == "__main__":
    # 1维输入 1维输出  100个隐藏节点
    p = FNN(1,1, 100)

    # 构造训练集 测试函数sin(x)
    x = np.array([i*0.005 for i in range(1000)]).reshape(1000,1)
    y = np.sin(x)

    # 训练模型
    p.fit(x,y)

    # 预测
    yy = p.predict(x)

    # 画图观察
    plt.plot(x,y)
    plt.plot(x,yy)
    plt.show()
