#Logistic Regression

import numpy as np
import theano 
import theano.tensor as T
import matplotlib
import matplotlib.pyplot as plt


rng=np.random

N=400 #训练数据的大小
feats=784 #输入变量的个数

#生成数据集D=(input,output)
D=(rng.randn(N,feats),rng.randint(size=N,low=0,high=2))
training_steps=10000

#定义theano符号变量
x=T.matrix('x')
y=T.vector('y')

#随机初始化权重向量w
#w和偏置b都是shared变量

w=theano.shared(rng.randn(feats),name='w')

#初始化偏置向量b
b=theano.shared(0.,name='b')

print("Initial model:")
print(w.get_value)
print(b.get_value)

#构造theano表达式图
p_1=1/(1+T.exp(-T.dot(x,w)-b)) #target=1的概率
prediction=p_1>0.5 #预测的阈值
xent=-y*T.log(p_1)-(1-y)*T.log(1-p_1) #交叉熵
cost=xent.mean()+0.01*(w**2).sum() #损失函数
gw,gb=T.grad(cost,[w,b])

train=theano.function(inputs=[x,y],
                      outputs=[prediction,xent],
                      updates=((w,w-0.1*gw),(b,b-0.1*gb)))
                    
predict=theano.function(inputs=[x],outputs=prediction)

fun=theano.functon(inputs=[x],outputs=p_1)

#训练
for i in range(training_steps):
    pred,err=train(D[0],D[1])
    
print("Final model")
print(w.get_value())
print(b.get_value())
print("target values for D:")
print(D[1])
print("prediction on D:")
print(predict(D[0]))
plt.scatter(D[0][:,0],D[1][:],c=D[1][:])
plt.show()