# -*- coding: utf-8 -*-
#MLP
import theano
import theano.tensor as T
import numpy as np
import timeit
from LogistRegress import LogisticRegression,load_data
import six.moves.cPickle as pickle

rng=np.random

#全连接层
class HiddenLayer(object):
    def __init__(self,rng,input,n_in,n_out,W=None,b=None,activation=T.tanh):
        self.input=input
        #对于tanh激活函数W用从sqrt(-6./(n_in+n_hidden))到sqrt(6.0/(n_in+n_hidden))的uniform采样初始化
        #uniform的输出用asarray转换成dtype为theano.config.floatX，这样代码就可以在GPU上运行
        #注意：选择初始化权重由激活函数决定，比如对于sigmoid激活函数建议是tanh的uniform结果的4倍
        if W is None:
            W_values=np.asarray(rng.uniform(low=-np.sqrt(6.0/(n_in+n_out)),
                                            high=np.sqrt(6.0/(n_in+n_out)),
                                            size=(n_in,n_out)),
                                dtype=theano.config.floatX)
                
            if activation==T.nnet.sigmoid:
                W_values*=4
            W=theano.shared(value=W_values,name='W',borrow=True)
        if b is None:
            b_values=np.zeros(n_out,dtype=theano.config.floatX)
            b=theano.shared(value=b_values,name='b',borrow=True)
            
        self.W=W
        self.b=b
        
        self.params=[self.W,self.b]
        #计算输出
        lin_output=T.dot(input,self.W)+self.b
        self.output=(lin_output if activation is None else activation(lin_output))

#多层感知器        
class MLP(object):
    def __init__(self,rng,input,n_in,n_hidden,n_out):
        self.hiddenLayer=HiddenLayer(rng=rng,input=input,n_in=n_in,n_out=n_hidden,activation=T.tanh)
        
        self.logRegressionLayer=LogisticRegression(input=self.hiddenLayer.output,n_in=n_hidden,n_out=n_out)
        
        #使用L1和L2正则化
        self.L1=(abs(self.hiddenLayer.W).sum()+abs(self.logRegressionLayer.W).sum())
        
        self.L2_sqr=((self.hiddenLayer.W**2).sum()+(self.logRegressionLayer.W**2).sum())
        
        #负log似然
        self.negative_log_likelihood=(self.logRegressionLayer.negative_log_likelihood)
        
        self.errors=self.logRegressionLayer.errors
        
        #参数
        self.params=self.hiddenLayer.params+self.logRegressionLayer.parms
        
        self.input=input
        
def test_mlp(learning_rate=0.01,L1_reg=0.00,L2_reg=0.0001,n_epochs=1000,dataset='mnist.pkl.gz',batch_size=50,n_hidden=500):
    datasets=load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    #build model
    print('...building the model')
    
    #index of minibatch
    index=T.lscalar('index')
    x=T.matrix('x')
    y=T.ivector('y')
    
    rng=np.random.RandomState(1234)
    
    classifier=MLP(rng=rng,input=x,n_in=28*28,n_hidden=n_hidden,n_out=10)
    
    #计算损失函数
    cost=(classifier.negative_log_likelihood(y)+L1_reg*classifier.L1+L2_reg*classifier.L2_sqr)
    
    #编译Theano函数
    test_model=theano.function(inputs=[index],
                               outputs=classifier.errors(y),
                               givens={
                                   x:test_set_x[index*batch_size:(index+1)*batch_size],
                                   y:test_set_y[index*batch_size:(index+1)*batch_size]
                               })
                               
    validate_model=theano.function(inputs=[index],
                                   outputs=classifier.errors(y),
                                   givens={
                                        x:valid_set_x[index*batch_size:(index+1)*batch_size],
                                        y:valid_set_y[index*batch_size:(index+1)*batch_size]
                                   })
    
    #计算梯度
    gparams=[T.grad(cost,param) for param in classifier.params]
    #计算updates
    updates=[(param,param-learning_rate*gparam) for param,gparam in zip(classifier.params,gparams)]
    
    train_model=theano.function(inputs=[index],
                                outputs=cost,
                                updates=updates,
                                givens={
                                    x:train_set_x[index*batch_size:(index+1)*batch_size],
                                    y:train_set_y[index*batch_size:(index+1)*batch_size]
                                })
    #Train Model#
    print('... train the model')
    #early-stopping参数
    patience=5000  #look as this many examples regardless
    patience_increase=2 #当找到一个新的最好的epoch后再等待patience_increse个epoch
    
    improvement_threshold=0.995 #有这么多的相对提高就看作是显著的提高
    #验证的频率
    validation_frequency=min(n_train_batches,patience/2)
    
    best_validation_loss=np.inf 
    test_score=0.
    start_time=timeit.default_timer()

    done_looping=False
    epoch=0
    while(epoch<n_epochs) and (not done_looping):
        epoch=epoch+1
        for minibatch_index in range(n_train_batches):
            #训练模型
            minibatch_avg_cost=train_model(minibatch_index)
            #迭代数
            it=(epoch-1)*n_train_batches+minibatch_index
            
            if(it+1)%validation_frequency==0:
                #计算验证集上的0-1loss
                validation_losses=[validate_model(i) for i in range(n_valid_batches)]
                this_validation_loss=np.mean(validation_losses)
                print('epoch %i, minibatch %i/%i,validation error %f %%' %(epoch,minibatch_index+1,n_train_batches,this_validation_loss))
                
                #如果目前是最好的score
                if this_validation_loss<best_validation_loss:
                    #如果loss改善明显就提高patience
                    if this_validation_loss<best_validation_loss*improvement_threshold:
                        patience=max(patience,it*patience_increase)
                    best_validation_loss=this_validation_loss
                    
                    #在测试集上测试
                    test_losses=[test_model(i) for i in range(n_test_batches)]
                    
                    test_score=np.mean(test_losses)
                    
                    print('epoch %i, minibatch %i/%i, test error of best model %f %%' %
                            (epoch,minibatch_index+1,n_train_batches,test_score*100))
                    
                    #保存最好的模型
                    #with open('best_model.pkl','wb') as f:
                        #pickle.dump(classifier,f)
                if patience<=it:
                    done_looping=True
                    break
    end_time=timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))

if __name__=='__main__':
    test_mlp()