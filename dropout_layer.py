def dropout_layer(state_before,use_noise,trng,keep_dim=0.5):
    '''使用二项分布随机选取结点,binomial为二项分布
    '''
    proj=T.swith(use_noise,
                     (state_before*trng.binomial(size=state_before.shape,p=keep_dim,n=1,dtype=state_before.dtype)),
                      state_before*keep_dim)

    return proj