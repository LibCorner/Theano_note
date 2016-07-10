# -*- coding: utf-8 -*-
#SimpleRNN
import theano
import theano.tensor as T
import numpy as np
from collections import OrderedDict

def sgd(cost, params, learning_rate):
    return [(param, param - learning_rate * T.grad(cost, param)) for param in params]


class SimpleRNN(object):
    def __init__(self,input_dim,hidden_dim,output_dim,activation=T.nnet.sigmoid):
        #输出层到隐层的权重
        self.Wx=theano.shared(name="Wx",
                              value=np.random.uniform(-1.0,1.0,(input_dim,hidden_dim))
                              .astype(theano.config.floatX))
        #隐层到隐层的权重
        self.Wh=theano.shared(name='Wh',
                              value=np.random.uniform(-1.0,1.0,(hidden_dim,hidden_dim))
                              .astype(theano.config.floatX))
        #隐层到输出层的权重                      
        self.Wo=theano.shared(name='Wo',
                              value=np.random.uniform(-1.0,1.0,(hidden_dim,output_dim))
                              .astype(theano.config.floatX))
        
        self.bh=theano.shared(name='bx',value=np.zeros(hidden_dim,dtype=theano.config.floatX))
        
        self.bo=theano.shared(name='bo',value=np.zeros(output_dim,dtype=theano.config.floatX))
        
        #初始的隐层状态
        self.h0=theano.shared(name='h0',value=np.zeros(hidden_dim,dtype=theano.config.floatX))
        
        self.params=[self.Wx,self.Wh,self.Wo,
                     self.bh,self.bo,self.h0]
                     
        self.activation=activation
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim
        self.recurrent()
                     
    def step(self,x_t,h_tm1):
        '''
            时间维上每输步输入的操作        
        '''
        h_t=self.activation(T.dot(x_t,self.Wx)+T.dot(h_tm1,self.Wh)+self.bh)
        
        s_t=T.nnet.softmax(T.dot(h_t,self.Wo)+self.bo)
        return [h_t,s_t]
        
    def recurrent(self):
        x=T.tensor3("x")
        y=T.ivector('y')   
        
        #scan的第一维总是时间维，通过dimshuffle把mini_batch的第一维变成时间维，第二维为batch维
        #使用T.alloc()实现动态大小的batch的初始隐层权重
        [h,s],_=theano.scan(fn=self.step,sequences=x.dimshuffle(1,0,2),outputs_info=[T.alloc(self.h0,x.shape[0],self.hidden_dim),None],n_steps=x.shape[1])
        
        p_y_given_x_sentence=s[:,0,:]
        y_pred=T.argmax(p_y_given_x_sentence,axis=1)
        
        #learning rate
        lr=T.scalar('lr')
        cost=-T.mean(T.log(p_y_given_x_sentence)[y])
        updates=sgd(cost,self.params,lr)
        
        self.classify=theano.function(inputs=[x],outputs=y_pred)
        
        self.train=theano.function(inputs=[x,y,lr],outputs=cost,updates=updates)
        
        self.predict=theano.function(inputs=[x],outputs=s[-1])
        

if __name__=='__main__':

   rnn=SimpleRNN(input_dim=4,hidden_dim=4,output_dim=4)
   x_train=np.array([[[1,1,1,1],[1,2,1,1]],
                     [[1,1,3,1],[1,2,3,4]],
                     [[1,2,3,1],[1,1,2,3]]])
   y_train=np.array([1,0,0])
   rnn.train(x_train,y_train,0.01)
        