# 牛顿法原理

$\beta^(t+1)=\beta^t-H^{-1}\nabla f(\beta^t)$

$\beta^t 是当前参数值，\nabla f(\beta^t)是目标函数的一阶导数，H：目标函数f(\beta)的海森(Hessian Matrix)矩阵（二阶导数矩阵，描述函数的曲率）$

为了更好的理解这个公式，现在降维到1维

泰勒series,函数在a点时的值可以近似为: $f(x) \approx f(a)+f'(a)(x-a)+\frac{1}{2}f''(a)(x-a)^2$

对这个系列，对X求导，得到 $f'(a)+f''(a)(x-a)=0$

求解得到 $x = a-\frac{f'(a)}{f''(a)}, 这也就是牛顿优化的更新公式x_{k+1} = x_k-\frac{f'(x_k)}{f''(x_k)} $

这个公式会自动朝着极值点移动，因为根据上面证明，它本身就是求导后设右边为0，也就是极值点。

同样的道理， $\beta^(t+1)=\beta^t-H^{-1}\nabla f(\beta^t)$ 这个公式其实和一维公式是一样的，只是把二阶导数在 $x_k$ 点的值替换成了Hessian矩阵，也就是多个变量的二阶导的矩阵，把一阶导数的值换成了向量。

## 通俗记忆

牛顿法就是随机初始参数，代入当前参数计算出Gradient向量和Hessian矩阵，最后用当前参数减去（Hessian的逆矩阵乘Gradient向量), $x^{k+1}=x^{k}-H^{-1}\nabla{g}$

# 逻辑回归例子

损失函数: $Loss = -\sum^N_{i=1}[y_ilog(p_i)+(1-y_i)log(1-p_i)]$，其中 $p=\frac{1}{1+e^{-(wx+b)}}$ 

![d](https://github.com/Tony980624/Newton-Raphson-method/blob/main/output3.png)

假如真实标签是1，预测的p也是1，（p的意思是x=1的概率），那么log(p) = log(1) = 0, 0*1 = 0,损失为0. 完全精准预测。

反之如果真实标签是1，预测的label=1的概率越低，那么就越不准确，log(p)的值越会越来越负，损失函数就越来越大

对损失函数求参数向量(w)的偏导函数： Gradient vector = $\sum^N_{i=1}[(p_i-y_i)x_i]$ 

梯度向量: $g = X^T(p-y)$ ， 注意$x_i$是数据向量，不是一个值， $p_i$也是一个向量，也就是带入参数后某一个函数点的预测值的向量

Hessian matrix: $X^TWX$ 

$$
W = \begin{bmatrix}
w_1 & 0 & 0 & \cdots & 0 \\
0 & w_2 & 0 & \cdots & 0 \\
0 & 0 & w_3 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & w_N
\end{bmatrix}
$$

其中：

$w_i = p_i (1 - p_i), \quad i = 1, 2, \dots, N$ ， 数据每一行的权重

# IRIS数据案例

```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# 读取数据
df = pd.read_csv(f'D:/Download/iris.csv')
# 标签是第五列
labels = df.iloc[:,4]
# 只保留两个品种，根据标签筛选
filtered_df = df[labels.isin(['setosa', 'versicolor'])]
# 提取训练数据
data = filtered_df.iloc[:,:4]
# 提取标签
labels = filtered_df.iloc[:,4]
# 把'setosa' 品种标签改为1， 另一个改为0
labels = labels.replace({'setosa': 1, 'versicolor': 0})
# columne bind 数据，把100*1个1和data列合并
data = np.c_[np.ones((len(data), 1)), data]
```

还剩下100行数据，这里为了简便就不分训练集和测试集了，直接同一个数据训练和测试

```
def sigmoid(z):
    # 返回预测的概率向量
    return 1 / (1 + np.exp(-z))

def newton(iterations, initial_para, data, labels):
    params = np.array(initial_para,dtype = float)  # [a, b, c, d, e]
    data = np.array(data)
    labels = np.array(labels)
    losses = []
    for _ in range(iterations):
        # 线性回归部分，数据矩阵乘参数向量 
        z = data @ params
        # 概率向量
        probabilities = sigmoid(z)
        # 记录损失函数的值
        loss = -np.sum(labels * np.log(probabilities) + (1 - labels) * np.log(1 - probabilities))
        losses.append(loss)
        errors = probabilities - labels  # (N,)
        # 梯度向量的公式，上面有写出
        g_vector = data.T @ errors  # (d,)
        # np.diag意思是扩充为对角矩阵，对角上每个值是每一行数据等于1的概率乘以等于0的概率
        W = np.diag(probabilities * (1 - probabilities))  # 对角矩阵 (N, N)
        Hessian = data.T @ W @ data  # (d, d)
        # 参数更新
        params -= np.linalg.inv(Hessian) @ g_vector
    return params,losses
```

## 查看损失函数

```
initial_para = [0, 0, 0, 0, 0]
params,loss = newton(iterations=10, initial_para=initial_para, data=data, labels=labels)
plt.plot(loss)
```

## 建立模型并预测

```
class logistic_mdel:
    def __init__(self,parameters):
        self.parameters = np.array(parameters)
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    def predict_proba(self,X):
        z = X@self.parameters
        return self.sigmoid(z)
    def predict(self,X,threshold = 0.5):
        probabilities = self.predict_proba(X)
        return (probabilities>=threshold).astype(int)
    
model = logistic_mdel(parameters=params)
proba = model.predict(data)
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(labels, proba)
print("Confusion Matrix:")
print(conf_matrix)

```

在训练集上True positive 和 True negative都全对

$$
Confusion Matrix: \begin{bmatrix}
50&0\\
0&50
\end{bmatrix}
$$
