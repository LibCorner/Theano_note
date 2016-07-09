# -*- coding: utf-8 -*-
#ScanPractise

import theano
import theano.tensor as T
import numpy as np

#累积results=A**k

A=T.matrix('a')
k=T.lscalar('k')
def muti(A,k):
    results,updates=theano.scan(fn=lambda prior_result,A:prior_result*A,outputs_info=T.ones_like(A),non_sequences=A,n_steps=k)
    
    r=results[-1]
    f=theano.function(inputs=[A,k],outputs=[r])
    return f

f=muti(A,k)
a=np.array([[2]])
k=5
out=f(a,k)
print out


'''
累加，从start加到end
'''
def a(prior_i,prior_result):
    return prior_i+1,prior_i+prior_result

#累加
def add(start,end,init):
    results,updates=theano.scan(fn=a,outputs_info=[start,init],n_steps=end-start+1)
    
    r=results[-1][-1]
    f=theano.function(inputs=[start,end,init],outputs=[r])
    return f
    
start=T.iscalar('s')
end=T.iscalar('e')
init=T.iscalar('i')
f=add(start,end,init)

r=f(1,100,0)
print r
