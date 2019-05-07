import  tensorflow as tf
def expand(x):
    c=tf.expand_dims(x,axis=4)
    return c

def softmax(a):
    a = tf.squeeze(input=a,axis=4)
    a=tf.negative(a)
    P = tf.nn.softmax(logits=a,axis=3)
    return P


def cost5(x):
    x_l=x[:,:,:,:,0]
    x_r=x[:,:,:,:,1]
    cost_vol = []
    cost_vol1 = []
    pad = [[0, 0], [0, 0], [0, 96], [0, 0]]
    cost = tf.pad(x_r, paddings=pad)
    for i in range(0, 96):
        slice = tf.slice(cost, [0, 0, i, 0], [-1, -1, 256, -1])
        cost_vol.append(slice)
        cost_vol1.append(x_l)

    cost_left = tf.stack(cost_vol1, axis=4)
    cost_right = tf.stack(cost_vol, axis=4)
    total1=tf.concat([cost_left,cost_right],axis=3)
    total1=tf.transpose(total1,[0,1,2,4,3])
    return total1

def softmax(x):
        x=tf.squeeze(x,axis=4)
        x=tf.negative(x)
        return x

def disp_accuracy(gt, pred):
    """Calculate the accuracy of correctly labeled pixels within a threshold."""
    pred = tf.cast(tf.argmax(pred, 3), tf.float32)
    gt = gt[:,:,:,0]
    tf.assert_equal(tf.rank(pred), tf.rank(gt))

    valid = tf.ones_like(gt, dtype=tf.int32)# tf.cast(tf.cond(tf.equal(tf.rank(gt), 3), lambda: gt[:,:,2], lambda: tf.ones_like(gt)), tf.int32)

    num_pixel = tf.reduce_sum(valid)
    num_correct = tf.reduce_sum(tf.cast(tf.less_equal(tf.abs(pred - gt), 3), tf.int32) * valid)

    return (num_correct / num_pixel) * 100

def def_loss(gt,pred):
    gt=gt[:,:,:,0]
    gt=tf.cast(gt,dtype=tf.int32)
    #pred=tf.clip_by_value(pred,1e-8,)
    loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt,logits=pred))
    #loss=(tf.clip_by_value(loss,1e-8,tf.reduce_max(loss)))*10
    return loss
