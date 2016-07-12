# -*- coding: utf-8 -*-
#LSTM
import theano
import theano.tensor as T
import numpy as np
from TERData import load_data

rng=np.random

def init_params(name,input_dim,output_dim):
    W=theano.shared(name=name,
                   value=np.asarray(rng.uniform(low=-6.0/(input_dim+output_dim),
                                                high=6.0/(input_dim+output_dim),
                                                size=(input_dim,output_dim)),
                                    dtype=theano.config.floatX))
    return W


class LSTM(object):
    def __init__(self,input,input_dim,hidden_dim,return_sequences=False):
        #输入和输出维数
        self.input=input.dimshuffle(1,0,2)
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.activation=T.nnet.sigmoid
        
        #初始化权值
        self.W_i=init_params('W_i',input_dim,hidden_dim)
        self.W_c=init_params('W_c',input_dim,hidden_dim)
        self.W_f=init_params('W_f',input_dim,hidden_dim)
        self.W_o=init_params('W_o',input_dim,hidden_dim)
        
        self.U_i=init_params('U_i',hidden_dim,hidden_dim)
        self.U_c=init_params('U_c',hidden_dim,hidden_dim)
        self.U_f=init_params('U_f',hidden_dim,hidden_dim)
        self.U_o=init_params('U_o',hidden_dim,hidden_dim)
        
        self.V_o=init_params('V_o',hidden_dim,hidden_dim)
        
        self.b_i=theano.shared(name='b_i',value=np.zeros(hidden_dim,dtype=theano.config.floatX))
        self.b_c=theano.shared(name='b_c',value=np.zeros(hidden_dim,dtype=theano.config.floatX))
        self.b_f=theano.shared(name='b_f',value=np.zeros(hidden_dim,dtype=theano.config.floatX))
        self.b_o=theano.shared(name='b_o',value=np.zeros(hidden_dim,dtype=theano.config.floatX))
        
        self.h0=theano.shared(name='h0',value=np.ones(hidden_dim,dtype=theano.config.floatX))
        self.c0=theano.shared(name='c0',value=np.ones(hidden_dim,dtype=theano.config.floatX))
        
        self.params=[self.W_c,self.W_f,self.W_i,self.W_o,
                     self.U_c,self.U_f,self.U_i,self.U_o,
                     self.b_c,self.b_f,self.b_i,self.b_o,
                     self.V_o,self.h0,self.c0]
        #计算输出
        output=self.recurrent()
        if return_sequences:
            self.output=output
        else:
            self.output=output[-1]
        
    def step(self,x_t,h_t,c_t):
        i_t=self.activation(T.dot(x_t,self.W_i)+T.dot(h_t,self.U_i)+self.b_i)
        C_i=T.tanh(T.dot(x_t,self.W_c)+T.dot(h_t,self.U_c)+self.b_c)
        
        f_t=self.activation(T.dot(x_t,self.W_f)+T.dot(h_t,self.U_f)+self.b_f)
        
        c=i_t*C_i+f_t*c_t
        
        o=self.activation(T.dot(x_t,self.W_o)+T.dot(h_t,self.U_o)+T.dot(c,self.V_o)+self.b_o)
        
        h=o*T.tanh(c)
        
        return h,c
        
    def recurrent(self):
        [h,s],_=theano.scan(fn=self.step,
                            sequences=self.input,
                            outputs_info=[T.alloc(self.h0,self.input.shape[1],self.hidden_dim),T.alloc(self.c0,self.input.shape[1],self.hidden_dim)],
                            n_steps=self.input.shape[0])
                            
        return h
        
def mse(true_y,pre_y):
    return T.mean(T.sqrt((true_y-pre_y)**2))

class RNNModel(object):
    def __init__(self,X_train,y_train,n_in=200,n_hidden=20,learning_rate=0.01,batch_size=20,objectve=mse):
        self.n_in=n_in
        self.n_hidden=n_hidden
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
        
        rnn=LSTM(input=x,
                     input_dim=self.n_in,
                     hidden_dim=self.n_hidden)
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
    X_train,y_train,bounds=load_data(samples_num=1000)
    rnn=RNNModel(X_train,y_train)
    rnn.fit()        