import tensorflow as tf

if __name__ == "__main__":
    a = tf.placeholder(tf.int32, [2, 3], name='Ta')
    b = tf.Variable(tf.truncated_normal([64, 200, 300, 1], stddev=0.3))
    te = tf.Variable(tf.truncated_normal([1148, 300], stddev=0.3))
    ta = tf.Variable(tf.ones([64, 300], tf.int32))
    c = tf.Variable(tf.truncated_normal([2, 100, 1, 100], stddev=0.3))
    convA = tf.nn.conv2d(b, c, strides=[1, 1, 1, 1], padding='VALID')
    tu = tf.nn.embedding_lookup(te, ta)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    #tu = sess.run(tu)
    convA = sess.run(convA)
    print(convA.shape)
    hA = tf.tanh(tf.squeeze(convA))
    print(hA.shape)
    tmphA = tf.reshape(hA, [64 * 299, 2000 // 2])

    #ha_mul_rand = tf.reshape(tf.matmul(tmphA, rand_matrix), [64, 300 - 1, 200 // 2])
    # r1 = tf.matmul(ha_mul_rand, hB, adjoint_b=True)
    # r3 = tf.matmul(ha_mul_rand, hNEG, adjoint_b=True)
    # att1 = tf.expand_dims(tf.stack(r1), -1)