def soft_argmin(val_model):\\
    """* Calculate soft-argmin loss for disparity to allow sub-pixel accuracy AND differentiability for backprop.\\
        *Soft-argmin is defined as the sum of each disparity, weighted by its normalized probability (softmax of negative cost of each disp).\\
    """
   *Take neg. cost for transforming into probability, remove last dim == 1\\
    prob = tf.negative(tf.squeeze(val_model, axis=-1))\\
    * Normalize probability\\
    with tf.name_scope('Softmax'):\\
    cost_vol_norm = tf.nn.softmax(prob, 1)\\
    d_max_p = cost_vol_norm.shape[1].value * res\\
    d_values = np.arange(0, 192, step=1, dtype=np.float32).reshape([1, 1, 1, -1])\\
    *Multiply disparity [0, D_max] with respective probability\\
    mult = tf.multiply(x=d_values, y=cost_vol_norm)\\
    *Sum over all disparities\\
    s_argmin = tf.reduce_sum(input_tensor=mult, axis=-1)\\
    return s_argmin\\
