import tensorflow as tf

# dcor with weight handling
def distance_corr_weighted(var_1, var_2, weight, power=1):
    """var_1: First variable to decorrelate (eg mass)
    var_2: Second variable to decorrelate (eg classifier output)
    weight: Per-example weight. (will be reweightes and used as normedweight where the sum of weights should add up to N (where N is the number of examples)
    power: Exponent used in calculating the distance correlation
    
    va1_1, var_2 and normedweight should all be 1D tf tensors with the same number of entries
    
    Usage: Add to your loss function. total_loss = BCE_loss + lambda * distance_corr
    """

    # normalise weights per batch
    weigth_batch_shape = tf.shape(weight)
    normedweight = weight/tf.math.reduce_sum(weight)*weigth_batch_shape[0].numpy()#*tensor_shape_list[0]

    # distance matrix ajk
    xx = tf.reshape(var_1, [-1, 1])
    xx = tf.tile(xx, [1, tf.size(var_1)])
    xx = tf.reshape(xx, [tf.size(var_1), tf.size(var_1)])

    yy = tf.transpose(xx)
    amat = tf.math.abs(xx-yy)

    # distance matrix bjk
    xx = tf.reshape(var_2, [-1, 1])
    xx = tf.tile(xx, [1, tf.size(var_2)])
    xx = tf.reshape(xx, [tf.size(var_2), tf.size(var_2)])
    
    yy = tf.transpose(xx)
    bmat = tf.math.abs(xx-yy)

    # calculate double centered distances
    amatavg = tf.reduce_mean(amat*normedweight, axis=1)
    bmatavg = tf.reduce_mean(bmat*normedweight, axis=1)

    minuend_1 = tf.tile(amatavg, [tf.size(var_1)])
    minuend_1 = tf.reshape(minuend_1, [tf.size(var_1), tf.size(var_1)])
    minuend_2 = tf.transpose(minuend_1)

    Amat = amat-minuend_1-minuend_2+tf.reduce_mean(amatavg*normedweight)


    minuend_1 = tf.tile(bmatavg, [tf.size(var_2)])
    minuend_1 = tf.reshape(minuend_1, [tf.size(var_2), tf.size(var_2)])
    minuend_2 = tf.transpose(minuend_1)

    Bmat = bmat-minuend_1-minuend_2+tf.reduce_mean(bmatavg*normedweight)

    # calculate covariance and variance 
    ABavg = tf.reduce_mean(Amat*Bmat*normedweight,axis=1)
    AAavg = tf.reduce_mean(Amat*Amat*normedweight,axis=1)
    BBavg = tf.reduce_mean(Bmat*Bmat*normedweight,axis=1)
   

   # calculate distance correlation   
    if power==1:
        dCorr = tf.reduce_mean(ABavg*normedweight)/tf.math.sqrt(tf.reduce_mean(AAavg*normedweight)*tf.reduce_mean(BBavg*normedweight))
    elif power==2:
        dCorr = (tf.reduce_mean(ABavg*normedweight))**2/(tf.reduce_mean(AAavg*normedweight)*tf.reduce_mean(BBavg*normedweight))
    else:
        dCorr = (tf.reduce_mean(ABavg*normedweight)/tf.math.sqrt(tf.reduce_mean(AAavg*normedweight)*tf.reduce_mean(BBavg*normedweight)))**power
  
    return dCorr



# same functions without weight handling
def distance_corr(var_1, var_2, power=2):
    """var_1: First variable to decorrelate (eg mass)
    var_2: Second variable to decorrelate (eg classifier output)
    normedweight: Per-example weight. Sum of weights should add up to N (where N is the number of examples)
    power: Exponent used in calculating the distance correlation
    
    va1_1, var_2 and normedweight should all be 1D tf tensors with the same number of entries
    
    Usage: Add to your loss function. total_loss = BCE_loss + lambda * distance_corr
    """

    # distance matrix ajk
    xx = tf.reshape(var_1, [-1, 1])
    xx = tf.tile(xx, [1, tf.size(var_1)])
    xx = tf.reshape(xx, [tf.size(var_1), tf.size(var_1)])

    yy = tf.transpose(xx)
    amat = tf.math.abs(xx-yy)

    # distance matrix bjk
    xx = tf.reshape(var_2, [-1, 1])
    xx = tf.tile(xx, [1, tf.size(var_2)])
    xx = tf.reshape(xx, [tf.size(var_2), tf.size(var_2)])
    
    yy = tf.transpose(xx)
    bmat = tf.math.abs(xx-yy)

    # calculate double centered distances
    amatavg = tf.reduce_mean(amat, axis=1)
    bmatavg = tf.reduce_mean(bmat, axis=1)

    minuend_1 = tf.tile(amatavg, [tf.size(var_1)])
    minuend_1 = tf.reshape(minuend_1, [tf.size(var_1), tf.size(var_1)])
    minuend_2 = tf.transpose(minuend_1)

    Amat = amat-minuend_1-minuend_2+tf.reduce_mean(amatavg)


    minuend_1 = tf.tile(bmatavg, [tf.size(var_2)])
    minuend_1 = tf.reshape(minuend_1, [tf.size(var_2), tf.size(var_2)])
    minuend_2 = tf.transpose(minuend_1)

    Bmat = bmat-minuend_1-minuend_2+tf.reduce_mean(bmatavg)

    # calculate covariance and variance 
    ABavg = tf.reduce_mean(Amat*Bmat,axis=1)
    AAavg = tf.reduce_mean(Amat*Amat,axis=1)
    BBavg = tf.reduce_mean(Bmat*Bmat,axis=1)

    # calculate distance correlation
    if power == 1:
        dCorr = tf.math.sqrt(tf.reduce_mean(ABavg))/tf.math.sqrt(tf.math.sqrt(tf.reduce_mean(AAavg))*tf.math.sqrt(tf.reduce_mean(BBavg)))
    elif power == 2:
        dCorr = tf.reduce_mean(ABavg)/tf.math.sqrt(tf.reduce_mean(AAavg)*tf.reduce_mean(BBavg))
    elif power==4:
        dCorr = (tf.reduce_mean(ABavg))**2/(tf.reduce_mean(AAavg)*tf.reduce_mean(BBavg))
    else:
        dCorr = (tf.reduce_mean(ABavg)/tf.math.sqrt(tf.reduce_mean(AAavg)*tf.reduce_mean(BBavg)))**power
  
    return dCorr
