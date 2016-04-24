#Theano Tutorial

#%matplotlib inline  #内联执行画图--如果不显示图形，在命令行中执行中

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import theano
#为了方便，tensor子模块加载为T
import theano.tensor as T

#1.基础

#1.1变量
#Theano所有的算法都是符号化定义的。它更像是在写数学而不是写代码。
#下面的Theano变量是符号，它们没有明确的值

#Theano中的theano.tensor子模块有简单的符号变量类型。
#这里我们定义一个scalar(0-d)变量
#参数给出了变量的名字
foo=T.scalar('foo')
#现在，我们定义另一个变量bar,它是foo的平方
bar=foo**2
#它也是一个theano变量
print(type(bar))
print(bar.type)
#使用theano的pp(pretty print)函数，我们可以看到
#bar是符号化的定义为foo的平方
print(theano.pp(bar))

#1.2函数
#要想使用theano来计算，需要定义一个符号函数，可以调用它传入真实的值来，得到计算结果

#使用foo和bar不能计算任何东西，需要先定义一个theano函数
#theano.function的第一个参数定义了函数的输入
#注意，bar依赖foo,所以这里foo是这个函数的输入
#theano.function会编译代码来根据给定的foo的值计算bar的值
f=theano.function([foo],bar)
print(f(3))

#有些情况下你可以用符号变量的eval方法来代替
#这比定义一个函数更方便
#eval方法传入一个字典，key是theano变量，value是该变量的值
print(bar.eval({foo:3}))

#我们也可以使用Python函数来构造Theano变量
#在这里它看起来有些迂腐，但是对于更加复杂的例子，它可以使语法更清晰
def square(x):
    return x**2
bar=square(foo)
print(bar.eval({foo:3}))

#1.3 theano.tensor
#Theano中也有向量、矩阵和张量(tensor)类型的变量，theano.tensor子模块有操作这些变量的方法
A=T.matrix('A')
x=T.vector('x')
b=T.vector('b')
y=T.dot(A,x)+b
#注意，矩阵的平方是矩阵里每个元素的平方
z=T.sum(A**2)
#theano.function可以一次计算多个式子
#也可以设置默认的参数值
#稍后会讲到theano.config.floatX
b_default=np.array([0,0],dtype=theano.config.floatX)
linear_mix=theano.function([A,x,theano.Param(b,default=b_default)],[y,z])
#Supplying values for A,x,and b
print(linear_mix(np.array([[1,2,3],
                          [4,5,6]],dtype=theano.config.floatX), #A
                          np.array([1,2,3],dtype=theano.config.floatX), #x
                          np.array([4,5],dtype=theano.config.floatX))) #b
#使用b的默认值
print(linear_mix(np.array([[1,2,3],[4,5,6]],dtype=theano.config.floatX),#A
                np.array([1,2,3],dtype=theano.config.floatX)))#x

#1.4 Shared变量
#Shared变量有一些不同，它们有明确的值，可以被set和get,可以被多个函数共享。
#因为它们有跨多个函数的状态，所以它也很有用
shared_var=theano.shared(np.array([[1,2],[3,4]],dtype=theano.config.floatX))
#shared变量的类型有它初始化的类型决定
print(shared_var.type())
#可以使用set_value方法来设置值
shared_var.set_value(np.array([[3,4],[2,1]],dtype=theano.config.floatX))
#使用get_value获取值
print(shared_var.get_value())

shared_squared=shared_var**2
#theano.functioin(inputs)告诉Theano要编译的函数的参数的值是什么
#需要注意的是，因为shared_var是共享的，它已经有一个值了，所以它不必作为函数的输入
#因此，隐式的把shared-var作为函数的一个输入，我们不用再输入中包含它
function_1=theano.function([],shared_squared)
print(function_1())

#1.4 updates
#shared变量可以使用theano.function的updates参数在函数里更新

#我们可以在函数里更新shared变量的值
subtract=T.matrix('subtract')
#updates传入一个字典(dict),key是shared变量，value是shared变量的新值
#这里，updates将设置shared_var=shared_var-subtract
function_2=theano.function([subtract],shared_var,updates={shared_var:shared_var-subtract})
print("shared_var before subtracting [[1,1],[1,1]] using function_2:")
print(shared_var.get_value())
#shared_var减去[[1,1],[1,1]]
function_2(np.array([[1,1],[1,1]],dtype=theano.config.floatX))
print("shared_var after calling function_2")
print(shared_var.get_value())
#注意，这也会改变function_1的输出，因为shared_var改变了
print("New output of function_1() (shared**2)")
print(function_1())

#1.5 Gradients
#使用theano的一个好处就是它有计算梯度的能力。
#这就允许你使用符号定义一个函数并快速的计算它的导数，而不用去实际的去求导。

#Recall that bar=foo**2
#我们可以计算bar对foo的梯度：
bar_grad=T.grad(bar,foo)
#我们希望bar_grad=2*foo
bar_grad.eval({foo:10})

#Recall that y=Ax+b
#我们也可以计算Jacobian(雅克比矩阵):
y_J=theano.gradient.jacobian(y,x)
linear_mix_J=theano.function([A,x,b],y_J)
#因为这是个线性mix,我们期望的输出是A
print(linear_mix_J(np.array([[9,8,7],[4,5,6]],dtype=theano.config.floatX),#A
                   np.array([1,2,3],dtype=theano.config.floatX),#x
                   np.array([4,5],dtype=theano.config.floatX))) #b
#我们也使用theano.gradient.hession计算Hessian（海森矩阵）                  
       
#1.6 Debugging
#在theano中调试有点难，因为实际运行的代码与你写的有很大差距。
#在编译函数之前来合理性检查你的theano表达式的一个简单方法是使用测试值

#创建另一个矩阵，“B”
B=T.matrix('B')
#创建一个符号变量表示A点乘B
#这时，theano不知道A的B的shape,所以它不知道A dot B是否合法
C=T.dot(A,B)
#现在，尝试使用它
#C.eval({A:np.zeros((3,4),dtype=theano.config.floatX),
#       B:np.zeros((5,6),dtype=theano.config.floatX)})
 #这样会报错
#当theano表达式很复杂时这种错误尤其令人困惑，
#它甚至不会告诉你A dot B发生在哪里，因为运行的代码不是你实际写的代码，而是编译的theano代码。
#幸运的是，“test values”可以规避这个问题，但是并不是所有的theano方法都允许test values

#这个告诉Theano我们将要使用test values,当他们有错的时候就发出警告
#设置的“warn” 意思是“当没有提供测试值时警告我”
theano.config.compute_test_value='warn'
#设置的tag.test_value属性给出了变量的测试值
A.tag.test_value=np.random.random((3,4)).astype(theano.config.floatX)
B.tag.test_value=np.random.random((5,6)).astype(theano.config.floatX)
#这样，我们当计算C的时候就会获得错误所在的行了
#C=T.dot(A,B)

#不需要使用test values就设置成off
theano.config.compute_test_value='off'

#另外一个debuging的用处是非法计算已经完成的情况。
#比如，结果是nan的计算，theano默认会允许nan的值被计算和使用，
#但是会对剩下的计算造成灾难
#我们可以在DebugeMode模式下编译theano,这就会使非法的计算导致一个错误。

#一个简单的除法函数
num=T.scalar("num")
den=T.scalar('den')
divide=theano.function([num,den],num/den)
print(divide(10,2))
#这个会产生错误,输出nan
print(divide(0,0))
#要使用debug mode,只要设置mode='DebugMode'
divide=theano.function([num,den],num/den,mode='DebugMode')
#NaNs now cause errors
#print(divide(0,0))
 
#2. 使用CPU vs GPU
#Theano 可以透明的编译到不同的硬件上，默认使用什么device由你的.theanorc文件和环境变量的定义来决定。  
#目前，当使用GPU时你应该使用float32，但是大多数人使用float64在CPU上。
#为了方便theano提供了floatX配置变量来指派使用什么精度的float。
#比如，你可以使用CPU的特定的环境变量集运行一个Python脚本。
# THEANO_FLAGS=device=cpu,floatX=float64 python your_script.py
#或者 GPU：
# THEANO_FLAGS=device=gpu,floatX=float32 python your_script.py

#你可以获取用来配置Theano的变量的值：
print(theano.config.device)
print(theano.config.floatX) 

#你也可以在运行时get/set
old_floatX=theano.config.floatX
theano.config.floatX='float32'
   
#谨慎的使用floatX
#比如，下面会使变量变成float64而不顾floatX的设置，因为numpy的默认类型：
var=theano.shared(np.array([1.3,2.4]))
print(var.type())#!!!
#所以，无论在哪里使用numpy数组，确保设置它的dtype为theano.config.floatX
var=theano.shared(np.array([1.3,2.4],dtype=theano.config.floatX))
print(var.type())
#Revet to old value
theano.config.floatX=old_floatX   

#3.例子：多层感知器MLP
#多层感知器的定义： http://en.wikipedia.org/wiki/Multilayer_perceptron
#我们将会使用数据点是列向量的convention(习俗)

#Layer 类
#我们会定义多层感知器为一系列的层，每一层连续的用输入产生网络的输出
#每一层定义为一个class,存储了权重矩阵和偏置向量以及计算该层输出的函数

class Layer(object):
    def __init__(self,W_init, b_init, activation):
        '''神经网络的一层，计算 s(Wx+b),这里s是非线性函数，x是输入向量。        
        
        ：参数：
            —— W_init：np.ndarray, shape=(n_output,n_input)
                初始化权值矩阵的值
            - b_init： np.ndarray,shape=(n_out,)
                初始化偏置向量
            - activation： theano.tensor.elemwise.Elemwise
                激活函数
        '''
        #根据W的初始化检索输入和输出的维数
        n_output,n_input=W_init.shape
        #确保b的大小是n_output
        assert b_init.shape==(n_output,)
        
        #所有的参数应该是shared变量
        #它们用来计算层的输出
        #当优化网络参数时会更新这些参数
        #注意，我们显示的要求W_init的类型是theano.config.floatX
        self.W=theano.shared(value=W_init.astype(theano.config.floatX),
                            #name参数是为了打印的目的
                            name='W',
                            #设置borrow=True允许Theano使用这个对象的用户内存，
                            #这样可以避免在构造上深度复制，代码稍微更快一些
                            #更多细节，查看http://deeplearning.net/software/theano/tutorial/aliasing.html
                            borrow=True
                             )
        #我们可以使用numpy的reshape方法的把偏置向量b转换成列向量
        #当b是列向量时，我们可以向网络层中传入一个矩阵形状的输入
        #并获得一个矩阵形状的输出
        self.b=theano.shared(value=b_init.reshape(n_output,1).astype(theano.config.floatX),
                             name='b',
                             borrow=True,
                             #Theano允许广播(broadcasting),与numpy相似
                             #然而，你需要显式的指明哪一维可以被广播
                             #通过设置broadcast=(False,True),可以指示b可以沿着它的第二维被广播（复制）
                             #这样就可以添加到另一个变量中。更多信息，查看
                             #http://deeplearning.net/software/theano/library/tensor/basic.html
                             broadcastable=(False,True))
                             
        self.activation = activation
        #计算损失函数对参数的梯度
        self.params=[self.W,self.b]

         
    def output(self,x):
        '''给一个输入，计算这层的输出
            ：参数：
                - x：theano.tensor.var.TensorVariavle
                    输入的Theano符号变量
            :return:
                - output: theano.tensor.var.TensorVariable
                    混合的，偏置的和激活的x
        '''
        #计算线性混合
        lin_output=T.dot(self.W,x)+self.b
        #如果没有激活参数输出仅仅是线性混合
        #否则，应用激活函数
        return (lin_output if self.activation is None else self.activation(lin_output))
            
#MLP class
#大部分的多层感知器的功能都在Layer class里；
#MLP class仅仅是Layers和它们参数的容器。
#output函数递归的计算每一层的输出。
#最后squared_error返回欧几里得距离误差。
#这个方法用来作为损失函数。
#这两个函数不是用来计算值的，而是用来创建可以计算值的函数的。
class MLP(object):
    def __init__(self,W_init,b_init,activations):
        '''
            多层感知器类，计算一些列层
            
            ：参数：
                - W_init: np.ndarray列表， len=N
                - b_init: np.ndarray列表，lne=N
                - activations: theano.tensor.elemwise.Elemwise列表， len=N
                    每一层输出的激活函数
        '''
        #确保所有的输入列表长度相同
        assert len(W_init)==len(b_init)==len(activations)
        
        #初始化layer列表
        self.layers=[]
        #构造layers
        for W,b,activation in zip(W_init,b_init,activations):
            self.layers.append(Layer(W,b,activation))
            
        #Combaine所有层的参数
        self.params=[]
        for layer in self.layers:
            self.params+=layer.params
    def output(self,x):
        '''
        计算多层感知器的输出
        :参数：
         - x：theano.tensor.var.TensorVarivable
              输入的theano符号变量
         :return:
             - output: theano.tensor.var.TensorVarivable
        '''
        #递归计算输出
        for layer in self.layers:
            x=layer.output(x)
        return x
    
    def squared_error(self,x,y):
        '''
        计算欧几里得误差
        :参数:
            - x: theano.tensor.var.TensorVarivable
            - y: theano.tensor.variavle for desired network output
        :return:
            -error: theano.tensor.var.TensorVariable
            误差
        '''
        return T.sum((self.output(x)-y)**2)
    
#梯度下降
def gradient_updates_momentum(cost,params,learning_rate,momentum):
    '''
    计算梯度
    ：参数：
        - cost: theano.tensor.var.TensorVariable
            要最小化的误差函数
        - params: list of theano.tensor.var.TensorVariable
            计算梯度的参数
        - learning_rate: float
            梯度下降的学习速率
        - momentum: float
            Momentum参数，至少为0，小于1
    ：return：
        updates: list
            每个参数的更新值
    '''
    #确保momentum是健壮的值
    assert momentum<1 and momentum>=0
    #更新列表
    updates=[]
    #对损失函数梯度下降
    for param in params:
        #对每个参数，我们会创建一个param_update的shared变量
        #这个变量会跟踪参数的每次迭代更新步骤
        #初始化它为0
        param_update=theano.shared(param.get_value()*0,broadcastable=param.broadcastable)
        #每个参数沿着梯度方向更新
        #然而，我们也根据给定的momentum值“mix in”混合之前步骤的值。
        #注意，当更新param_update时，我们使用old值和新的梯度step.
        updates.append((param,param-learning_rate*param_update))
        #注意，我们不需要反向传播求导来计算updates, 只需要使用T.grad
        updates.append((param_update,momentum*param_update+(1. -momentum)*T.grad(cost,param)))
    return updates
    
# Toy example
#训练神经网络来划分二维高斯分布的点
np.random.seed(0)
#点的个数
N=1000
#每个簇的标签
y=np.random.random_integers(0,1,N)
#每个簇的平均值
means=np.array([[-1,1],[-1,1]])
#每个簇在X和Y方向的协方差
covariances=np.random.random_sample((2,2))+1
#每个点的维度
X=np.vstack([np.random.randn(N)*covariances[0,y]+means[0,y],
             np.random.randn(N)*covariances[1,y]+means[1,y]]).astype(theano.config.floatX)
             
#Convert to targets, as floatX
y=y.astype(theano.config.floatX)
#Plot the data
plt.figure(figsize=(8,8))
plt.scatter(X[0,:],X[1,:],c=y,lw=.3,s=3,cmap=plt.cm.cool)
plt.axis([-6,6,-6,6])
plt.show()

#首先，设置每层的大小和层数
#输入层的大小是训练数据的维数
# 输出大小是一维的，类标签：0或1
# 最后，让隐层是输出的两倍
# 如果我们要更多的层，可以在这个list里添加另外的层的大小
layer_sizes = [X.shape[0], X.shape[0]*2, 1]
# 初始化参数的值
W_init = []
b_init = []
activations = []
for n_input, n_output in zip(layer_sizes[:-1], layer_sizes[1:]):
    # Getting the correct initialization matters a lot for non-toy problems.
    # However, here we can just use the following initialization with success:
    # Normally distribute initial weights
    W_init.append(np.random.randn(n_output, n_input))
    # Set initial biases to 1
    b_init.append(np.ones(n_output))
    # We'll use sigmoid activation for all layers
    # Note that this doesn't make a ton of sense when using squared distance
    # because the sigmoid function is bounded on [0, 1].
    activations.append(T.nnet.sigmoid)
# 创建MLP类的实例
mlp = MLP(W_init, b_init, activations)

# 为MLP的输入创建theano变量
mlp_input = T.matrix('mlp_input')
# ... and the desired output
mlp_target = T.vector('mlp_target')
# Learning rate and momentum hyperparameter values
# Again, for non-toy problems these values can make a big difference
# as to whether the network (quickly) converges on a good local minimum.
learning_rate = 0.01
momentum = 0.9
# Create a function for computing the cost of the network given an input
cost = mlp.squared_error(mlp_input, mlp_target)
# Create a theano function for training the network
train = theano.function([mlp_input, mlp_target], cost,
                        updates=gradient_updates_momentum(cost, mlp.params, learning_rate, momentum))
# Create a theano function for computing the MLP's output given some input
mlp_output = theano.function([mlp_input], mlp.output(mlp_input))

# Keep track of the number of training iterations performed
iteration = 0
# We'll only train the network with 20 iterations.
# A more common technique is to use a hold-out validation set.
# When the validation error starts to increase, the network is overfitting,
# so we stop training the net.  This is called "early stopping", which we won't do here.
max_iteration = 20
while iteration < max_iteration:
    # Train the network using the entire training set.
    # With large datasets, it's much more common to use stochastic or mini-batch gradient descent
    # where only a subset (or a single point) of the training set is used at each iteration.
    # This can also help the network to avoid local minima.
    current_cost = train(X, y)
    # Get the current network output for all points in the training set
    current_output = mlp_output(X)
    # We can compute the accuracy by thresholding the output
    # and computing the proportion of points whose class match the ground truth class.
    accuracy = np.mean((current_output > .5) == y)
    # Plot network output after this iteration
    plt.figure(figsize=(8, 8))
    plt.scatter(X[0, :], X[1, :], c=current_output,
                lw=.3, s=3, cmap=plt.cm.cool, vmin=0, vmax=1)
    plt.axis([-6, 6, -6, 6])
    plt.title('Cost: {:.3f}, Accuracy: {:.3f}'.format(float(current_cost), accuracy))
    plt.show()
    iteration += 1
    
           