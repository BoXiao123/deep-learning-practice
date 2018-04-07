import tensorflow as tf
def sapce_to_deth_x2(x):
    return tf.space_to_depth(x,block_size=2)
