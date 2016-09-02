## Condition: 条件

1. theano.tensor.switch(cond,ift,iff):
    * cond: 条件
    * ift: 符号张量，如果true,返回该值。
    * iff: 符号张量。如果false,返回该值。
2. theano.tensor.where(cond,ift,iff):
    * 与switc一样。
3. tensor.tensor.clip(x,min,max):
    * 比较x中每个元素与min,max的大小，如果大于max,该元素值为max,
    * 如果小于min，该元素值为min。
    * 返回最后得到的x。