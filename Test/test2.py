import tensorflow as tf

if __name__ == "__main__":
    Text_a = tf.Variable(tf.ones([64, 300], tf.int32))
    text_embed = tf.Variable(tf.truncated_normal([50, 100], stddev=0.3))
    TA = tf.nn.embedding_lookup(text_embed, Text_a)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print(TA.shape)
    T_A = tf.expand_dims(TA, -1)
    print(T_A.shape)

    W2 = tf.Variable(tf.truncated_normal([2, 100, 1, 100], stddev=0.3))
    rand_matrix = tf.Variable(tf.truncated_normal([100, 100], stddev=0.3))

    convA = tf.nn.conv2d(T_A, W2, strides=[1, 1, 1, 1], padding='VALID')
    print(convA.shape)
    hA = tf.tanh(tf.squeeze(convA))
    print(hA.shape)
    tmphA = tf.reshape(hA, [64 * (300 - 1), 100])
    print(tmphA.shape)
    ha_mul_rand = tf.reshape(tf.matmul(tmphA, rand_matrix),
                             [64, 300 - 1, 100])
    print(ha_mul_rand.shape)
    hB = tf.tanh(tf.squeeze(convA))
    print(hB.shape)
    r1 = tf.matmul(ha_mul_rand, hB, adjoint_b=True)
    print(r1.shape)
    att1 = tf.expand_dims(tf.stack(r1), -1)
    print(att1.shape)
    att1 = tf.tanh(att1)
    pooled_A = tf.reduce_mean(att1, 2)
    print(pooled_A.shape)
    pooled_B = tf.reduce_mean(att1, 1)
    print(pooled_B.shape)

    a_flat = tf.squeeze(pooled_A)
    b_flat = tf.squeeze(pooled_B)
 #   print(a_flat.shape)

    w_A = tf.nn.softmax(a_flat)
    print(w_A.shape)

    rep_A = tf.expand_dims(w_A, -1)
 #   print(rep_A.shape)
    hA = tf.transpose(hA, perm=[0, 2, 1])
#    print(hA.shape)

    rep1 = tf.matmul(hA, rep_A)

 #   print(rep1.shape)
    attA = tf.squeeze(rep1)
 #   print(attA.shape)