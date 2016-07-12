# -*- coding: utf-8 -*-
import theano
import theano.tensor as T
import numpy as np
from TERData import load_data

rng=np.random

class RNNLayer(object):
    def __init__(self,input,input_dim,hidden_dim,output_dim,return_sequences=False):
        '''
            1.首先确定输入和输出的维数
            2.然后，定义网络权重
            3.再写出符号计算过程
            4.最后得到符号输出
        '''
        self.input=input.dimshuffle(1,0,2)
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim
        self.activation=T.nnet.sigmoid
        
        rng=np.random
        #初始化网络中的参数
        self.W_in=theano.shared(name="W_in",
                                value=np.asarray(rng.uniform(low=-6.0/(input_dim+hidden_dim),
                                                             high=6.0/(input_dim+hidden_dim),
                                                             size=(input_dim,hidden_dim)),
                                                 dtype=theano.config.floatX))
        self.W_h=theano.shared(name="W_h",
                                value=np.asarray(rng.uniform(low=-6.0/(hidden_dim+hidden_dim),
                                                             high=6.0/(hidden_dim+hidden_dim),
                                                             size=(hidden_dim,hidden_dim)),
                                                 dtype=theano.config.floatX))
        self.W_out=theano.shared(name="W_out",
                                value=np.asarray(rng.uniform(low=-6.0/(hidden_dim+output_dim),
                                                             high=6.0/(hidden_dim+output_dim),
                                                             size=(hidden_dim,output_dim)),
                                                 dtype=theano.config.floatX))
                                                 
        self.b_h=theano.shared(name='b_h',value=np.zeros(hidden_dim,dtype=theano.config.floatX))
        
        self.b_out=theano.shared(name='b_out',value=np.zeros(output_dim,dtype=theano.config.floatX))
        
        #初始的隐层状态
        self.h0=theano.shared(np.zeros(hidden_dim,dtype=theano.config.floatX))
        
        self.params=[self.W_in,self.W_h,self.W_out,
                     self.b_h,self.b_out,self.h0]
        
        #计算输出
        out=self.recurrent()
        if return_sequences:
            self.output=out
        else:
            self.output=out[-1]
        
    def step(self,x_t,h_tm1):
        '''
            rnn每次迭代的操作        
        '''
        h_t=self.activation(T.dot(x_t,self.W_in)+T.dot(h_tm1,self.W_h)+self.b_h)
        s_t=T.nnet.softmax(T.dot(h_t,self.W_out)+self.b_out)
        
        return [h_t,s_t]
    
    def recurrent(self):
        '''
            scan函数对输入序列进行迭代,
            输入序列的第一维是时间维，第二维是batch维,
            使用T.alloc()动态的指定初始隐层状态的大小。
        '''
        [h,s],_=theano.scan(fn=self.step,sequences=self.input,outputs_info=[T.alloc(self.h0,self.input.shape[1],self.hidden_dim),None],n_steps=self.input.shape[0])
        
        return s

def mse(true_y,pre_y):
    return T.mean(T.sqrt((true_y-pre_y)**2))

class RNNModel(object):
    def __init__(self,X_train,y_train,n_in=200,n_hidden=200,n_out=20,learning_rate=0.01,batch_size=100,objectve=mse):
        self.n_in=n_in
        self.n_hidden=n_hidden
        self.n_out=n_out
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.objective=objectve
        
        self.x,self.y=self.shared_dataset(X_train,y_train)
        #总样本数
        n_x=self.x.get_value(borrow=True).shape[0]
        self.batches=n_x//batch_size
        
        self.build(self.x,self.y)
        
        
    def shared_dataset(self,datas,labels,borrow=True):
        '''
            把训练数据转换成shared变量        
        '''
        shared_x=theano.shared(np.asarray(datas,dtype=theano.config.floatX),
                               borrow=borrow)
        shared_y=theano.shared(np.asarray(labels,dtype=theano.config.floatX),
                               borrow=borrow)
                
        return shared_x,shared_y
        
    def build(self,train_set_x,train_set_y):
        index=T.lscalar('index') #index of minibatch
        x=T.tensor3('x')
        y=T.matrix('y')
        
        rnn=RNNLayer(input=x,
                     input_dim=self.n_in,
                     hidden_dim=self.n_hidden,
                     output_dim=self.n_out)
        output=rnn.output 
        
        #计算误差
        loss=self.objective(y,output)
        
        params=rnn.params
        #计算梯度
        grads=T.grad(loss,params)
        
        updates=[(param_i,param_i-self.learning_rate*grad_i) for param_i,grad_i in zip(params,grads)]
        
        self.train_model=theano.function(inputs=[index],outputs=loss,updates=updates,
                                    givens={x:train_set_x[index*self.batch_size:(index+1)*self.batch_size],
                                            y:train_set_y[index*self.batch_size:(index+1)*self.batch_size]})
        self.predict=theano.function(inputs=[x],outputs=output)
     
    def fit(self,n_epochs=10):
        for i in range(n_epochs):
            loss=0
            print 'epoch',i
            for index in range(self.batches):
                loss+=self.train_model(index)
            print 'loss',loss
if __name__=="__main__":
    X_train,y_train,bounds=load_data()
    rnn=RNNModel(X_train,y_train)
    rnn.fit()
    