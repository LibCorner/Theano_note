```python
def scan(fn,
         sequences=None,
         outputs_info=None,
         non_sequences=None,
         n_steps=None,
         truncate_gradient=-1,
         go_backwards=False,
         mode=None,
         name=None,
         profile=False,
         allow_gc=None,
         strict=False):
```
参数：
------------------------
## fn
`
``fn``是一个函数，描述了在一步``scan``中的操作。该函数的输入是表示输入序列的theano变量和之前步骤的输出值，以及scan的其他的参数，如``non_sequences``。
scan把这些参数传递给``fn``的顺序如下：
`

* 第一个序列的所有的time slices 
* 第二个序列的所有的time slices
* ...
* 最后一个序列的所有的time slices
* 第一个输出的所有past slices
* 第二个输出的所有的 past slices
* ...
* 最后一个输出的所有past slices
* 其他的参数（`non_sequences`里的参数）

`
序列的顺序与`sequences`里给序列顺序一致。输出的顺序与`outputs_info`里的顺序一致。对于任意的序列或输出，time slices的顺序与给定的`taps`顺序一致。
比如，下面的代码：
`
```python
   scan(fn,sequences=[dict(input=Sequences1,taps=[-3,2,-1]),
                        Sequences2,
                        dict(input=Sequence3,taps=3)],
            outputs_info=[dict(initial=Output1,taps=[-3,-5]),
                          dict(initial=Output2,taps=None),
                          Output3],
            non_sequences=[Argument1,Argument2])
```


``fn``按照taps给定的顺序应该参数的顺序应该如下：

* ``Sequence1[t-3]``
* ``Sequence1[t+2]``
* ``Sequence1[t-1]``
* ``Sequence2[t]``
* ``Sequence3[t+3]``
* ``Output1[t-3]``
* ``Output1[t-5]``
* ``Output3[t-1]``
* ``Argument1``
* ``Argument2``

`
``non_sequences``列表也可以包含shared变量, scan也可以根据它自己的来计算，因此这些可以省略。在一定程度上，scan也可以计算其它的``non_sequcences``(not shared)
即使不传递给scan（但是被`fn`使用）。一个简单的例子如下：
`
```python
import theano.tensor as TT
W=TT.matrix()
W_2=W**2
def f(x):
   return TT.dot(x,W_2)
```
`
这个函数期望返回两个东西，一个是outputs列表，按照`outputs_info` 的顺序。第二，`fn`应该返回一个update字典,描述每步迭代如何更新shared变量。
这两个列表的顺序没有限制，`fn`即可以返回`（outputs_list,update_dictionary）`也可以返回`(update_dictionary,outputs_list)`,或者只是两个中的一个。
`

`
要使用`scan`作为while循环，用户需要改变函数`fn`, 返回一个停止条件。要这样做，就要把条件包裹在一个`until`类中。这个条件应该作为第三个元素返回，例如：
`

```python
...
return [y1_t,y2_t],{x:x+1},theano.scan_module.until(x<50)
```
`
需要注意的是，即使传给scan了一个条件，steps参数也要设置一个数字。
`

## sequences
----------------------
`sequences`是Theano变量的列表或字典，描述了`scan`函数要遍历迭代的序列。如果sequence在字典中，可以提供一些关于序列的可选的操作信息。字典应该有如下的keys:

* `input` (*mandatory*)  -- Theano 变量，表示输入序列。
* `taps`  -- `fn`需要的序列的时间taps，它们是一个整数列表，值`k`表示在第`t`次迭代scan会把slice(切片)`t+k`传递给`fn`,默认值是`[0]`。

`
list里的任何的Theano变量都会自动的包装成`taps`为`[0]`的dictionary
`

## outputs_info
---------------------
`outputs_info` 是Theano变量列表或字典，描述了循环计算的outputs的初始状态。当这个初始状态作为字典时，可以为与初始状态相应的output提供一些可选的信息。字典应该有
如下的keys:

* `initial` -- Theano变量，表示给定的output的初始状态。如果，ouput不用被递归的计算或者不需要初始状态，这个可以省略。假如前面time step的output被`fn`使用，初始状态应该与output的shape一致，并且应该没有output数据类型的下转型。如果使用了多个time taps初始状态应该有一个额外的维，覆盖所有的taps.比如，如果
使用`-5`,`-2`和`-1`作为past taps，在第0步，`fn`需要`output[-5]`,`output[-2]`和`output[-1]`,这些应该有初始状态给定，shape应该是(5,)+output.shape。如果这个变量
包含的初始状态名为`init_y`，那么`init_y[0]`对应`output[-5]`,`init_y[1]`对应`output[-4]`,`init_y[2]`对应`output[-3]`,`init_y[4]`对应`output[-1]`。
* `taps` --输出的Temporal taps，会传递给`fn`。它们是*负整数*的列表，值`k`表示在第`t`步scan会传给`fn`slice`t+k`。


`
  ``scan`` will follow this logic if partial information is given:

        * If an output is not wrapped in a dictionary, ``scan`` will wrap
          it in one assuming that you use only the last step of the output
          (i.e. it makes your tap value list equal to [-1]).
        * If you wrap an output in a dictionary and you do not provide any
          taps but you provide an initial state it will assume that you are
          using only a tap value of -1.
        * If you wrap an output in a dictionary but you do not provide any
          initial state, it assumes that you are not using any form of
          taps.
        * If you provide a ``None`` instead of a variable or a empty
          dictionary ``scan`` assumes that you will not use any taps for
          this output (like for example in case of a map)

        If ``outputs_info`` is an empty list or None, ``scan`` assumes
        that no tap is used for any of the outputs. If information is
        provided just for a subset of the outputs an exception is
        raised (because there is no convention on how scan should map
        the provided information to the outputs of ``fn``)
`

## non_sequences
--------------------------
`non_sequences`是每步迭代传给`fn`的参数列表。

## n_steps
--------------------------
`non_sequences`是迭代次数，int或Theano scalar类型。如果输入序列没有足够多的元素，scan会抛出错误。如果值是0scan会输出0行；如果值是负数scan会在time上反向运行。
如果`go_backwoards`已经设置了并且`n_steps`也是negative, scan就会正向运行。如果`n_step`没有设置scan会根据输入序列计算出迭代的次数。

## truncate_gradient
------------------------
`truncate_gradient`是在truncated BPTT使用的steps数。如果通过scan操作计算梯度，就会使用backpropagation through time计算。通过提供不同的值，你可以选择使用truncated BPTT代替典型的BPTT,这样就只会在时间维上反向传播`truncate_gradient`次。

## go_backwords
-------------------------
    go_backwards
        ``go_backwards`` is a flag indicating if ``scan`` should go
        backwards through the sequences. If you think of each sequence
        as indexed by time, making this flag True would mean that
        ``scan`` goes back in time, namely that for any sequence it
        starts from the end and goes towards 0.

    name
        When profiling ``scan``, it is crucial to provide a name for any
        instance of ``scan``. The profiler will produce an overall
        profile of your code as well as profiles for the computation of
        one step of each instance of ``scan``. The ``name`` of the instance
        appears in those profiles and can greatly help to disambiguate
        information.

    mode
        It is recommended to leave this argument to None, especially
        when profiling ``scan`` (otherwise the results are not going to
        be accurate). If you prefer the computations of one step of
        ``scan`` to be done differently then the entire function, you
        can use this parameter to describe how the computations in this
        loop are done (see ``theano.function`` for details about
        possible values and their meaning).

    profile
        Flag or string. If true, or different from the empty string, a
        profile object will be created and attached to the inner graph of
        scan. In case ``profile`` is True, the profile object will have the
        name of the scan instance, otherwise it will have the passed string.
        Profile object collect (and print) information only when running the
        inner graph with the new cvm linker ( with default modes,
        other linkers this argument is useless)

    allow_gc
        Set the value of allow gc for the internal graph of scan.  If
        set to None, this will use the value of config.scan.allow_gc.

    strict
        If true, all the shared variables used in ``fn`` must be provided as a
        part of ``non_sequences`` or ``sequences``.

## Returns
---------------
元组

`
返回元组的形式为(outputs,updates); `outputs`是Theano变量或Theano变量的列表，表示`scan`的输出（与`outputs_inf`）的顺序一致。
`updates`是字典的子类，具体化了scan中使用的shared变量的updates规则。
这个字典应该传递给`theano.function`。与标准的字典的不同是，我们验证了keys是SharedVariable，ddition of those dictionary are validated to be consistent.
`