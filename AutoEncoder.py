# -*- coding: utf-8 -*-
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
#AutoEncoder
class dA(object):
    def __init__(self,numpy_rng,theano_rng=None,input=None,n_visible=784,n_hidden=500,
                 W=None,bhid=None,bvis=None):
        '''
            初始化类，具体化可视单元的个数（输入的维度d），隐藏单元的个数（隐藏层的维度）和
            corruption level。构造方法也接受输入、权重和偏执的符号变量。这样的符号变量很有用，
            比如当输入是某个计算的结果或当权重在dA层和MLP层之间共享时。当处理SdAs这就经常发生，
            在层2的dA的输入是层1的dA的输出，以及dA的权重用在第二阶段的训练来构造一个MLP.
            :type numpy_rng:numpy.random.RandomState
            :param numpy_rng: 随机数生成器用来生成权重
            
            :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
            :param theano_rng: theano随机生成器;如果给None就基于‘rng’里的seed来生成。
            
            :type input: theano.tensor.TensorType
            :param input: 输入的符号描述,或None for standalone dA
            
            :type n_visible: int
            :param n_visible: 可视单元的个数
            
            :type n_hidden: int
            :param n_hidden: 隐藏层的单元个数
            
            :type W: theano.tensor.TensorType
            :param W:Theano变量存放在dA层和其他结构中共享的权重；如果dA是standalone就把这个设为None
            
            :type bvis:theano.tensor.TensorType
            :param bvis:Theano变量存放偏置值
        '''
        self.n_visible=n_visible
        self.n_hidden=n_hidden
        
        #创建一个Theano的随机生成器生成符号随机值
        if not theano_rng:
            theano_rng=RandomStreams(numpy_rng.randint(2**30))
        #注意：W'写成'W_prime', b为'b_prime'
        if not W:
            #W 使用'initial_W'来初始化，            
            #'initial_W'是从-4*sart(6./(n_visible+n_hidden))和
            #4*sqrt(6./n_hidden+n_visible) 之间采样
            initial_W=np.asarray(
                numpy_rng.uniform(low=-4*np.sqrt(6./(n_hidden+n_visible)),
                                  high=4*np.sqrt(6./(n_hidden+n_visible)),
                                  size=(n_visible,n_hidden)
                                  ),dtype=theano.config.floatX)
            W=theano.shared(value=initial_W,name='W',borrow=True)
        if not bvis:
            bvis=theano.shared(value=np.zeros(n_visible,dtype=theano.config.floatX),borrow=True)
        if not bhid:
            bhid=theano.shared(value=np.zeros(n_hidden,dtype=theano.config.floatX),
                               name='b',borrow=True)
        self.W=W
        #b对应隐藏层的bias
        self.b=bhid
        #b_prime 对应可视层的bias
        self.b_prime=bvis
        #tied weights,W_prime是W的转置
        self.W_prime=self.W.T
        self.theano_rng=theano_rng
        #如果没有给输入就生成一个变量表示输入
        if input is None:
            #我们使用matrix，因为我们希望输入的是minibatch个样本，每个样本一行
            self.x=T.dmatrix(name='input')
        else:
            self.x=input
        self.params=[self.W,self.b,self.b_prime]
    
    def get_hidden_values(self,input):
        '''计算隐藏层的值'''
        return T.nnet.sigmoid(T.dot(input,self.W)+self.b)
    def get_reconstructed_input(self,hidden):
        '''根据隐藏层的值计算重新构造的输入'''
        return T.nnet.sigmoid(T.dot(hidden,self.W_prime)+self.b_prime)
    def get_corrupted_input(self,input,corruption_level):
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input
        
    def get_cost_updates(self,corruption_level,learning_rate):
        '''这个函数计算cost和updates'''
        tilde_x=self.get_corrupted_input(self.x,corruption_level)
        y=self.get_hidden_values(tilde_x)
        z=self.get_reconstructed_input(y)
        #注意：我们计算数据点的和，如果使用的是minibatches,L就是一个向量
        #每一维是minibatch中的一个样本
        L=-T.sum(self.x*T.log(z)+(1-self.x)*T.log(1-z),axis=1)
        #注意：L现在是一个向量，每个元素是一个交叉熵cost,我们需要计算所有的这些元素
        #的平均来获取minibatch的cost
        cost=T.mean(L)
        #计算cost对参数的梯度
        gparams=T.grad(cost,self.params)
        #生成updates
        updates=[(param,param-learning_rate*gparam) for param,gparam in zip(self.params,gparams)]
        
        return (cost,updates)


def train(train_set_x,training_epochs=10,batch_size=20,learning_rate=0.1,input_dim=10,hidden_dim=20):
    index=T.lscalar() #minibatch的index
    x=T.matrix('x') #输入是栅格化的图像
    #把输入数据转换成shared
    train_set_x=theano.shared(value=np.asarray(train_set_x,dtype=theano.config.floatX))
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size 
    rng=np.random.RandomState(123)
    theano_rng=RandomStreams(rng.randint(2**30))
    
    da=dA(numpy_rng=rng,theano_rng=theano_rng,input=x,n_visible=input_dim,n_hidden=hidden_dim)
    
    cost,updates=da.get_cost_updates(corruption_level=0.3,learning_rate=learning_rate)        
    
    train_da=theano.function([index],cost,updates=updates,
                             givens={
                             x:train_set_x[index*batch_size:(index+1)*batch_size]})
    ###############
    #Training     #
    ###############
    #便利训练epoch
    for epoch in range(training_epochs):                             
        #遍历训练集
        c=[]
        for batch_index in range(n_train_batches):
            c.append(train_da(batch_index))
        print('Training epoch %d,cost'%epoch,np.mean(c))
data=np.random.random((100,10))
train(data)
