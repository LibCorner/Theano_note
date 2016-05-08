import theano
import theano.tensor as T
import numpy

x,y,b = T.dvectors('x','y','b')
W = T.dmatrix('W')
y = T.nnet.softmax(T.dot(W,x) + b)
out=theano.function([W,x,b],y)

#softmax函数
x=T.dmatrix('x')
y=T.nnet.softmax(x)
softmax=theano.function([x],y)

#concat函数
x=T.vector('x')
y=T.vector('y')
i=T.iscalar('i')
y1=T.concatenate([x,y],axis=i)
concat=theano.function([x,y,i],y1)

#cos
x=T.matrix('x')
y=T.matrix('y')
out=T.sum(x*y,axis=-1)/(T.sqrt(T.sum(x*x,axis=-1)*T.sum(y*y,axis=-1))+0.0000000000001)
cos=theano.function([x,y],out)


x=[1,2]
b=[1,2]
W=[[1,1],[1,2]]


