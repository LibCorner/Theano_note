def dropout_layer(state_before,use_noise,trng,keep_dim=0.5):
    '''ʹ�ö���ֲ����ѡȡ���,binomialΪ����ֲ�
    '''
    proj=T.swith(use_noise,
                     (state_before*trng.binomial(size=state_before.shape,p=keep_dim,n=1,dtype=state_before.dtype)),
                      state_before*keep_dim)

    return proj