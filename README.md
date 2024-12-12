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

损失函数: $Loss = -\sum^N_{i=1}[y_ilog(p_i)+(1-y_i)log(1-p_i)]$

![d](https://github.com/Tony980624/Newton-Raphson-method/blob/main/output1.png)

假如真实标签是1，预测的p也是1，（p的意思是x=1的概率），那么log(p) = log(1) = 0, 0*1 = 0,损失为0. 完全精准预测。

反之如果真实标签是1，预测的label=1的概率越低，那么就越不准确，log(p)的值越会越来越负，损失函数就越来越大
