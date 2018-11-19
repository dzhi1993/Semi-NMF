from __future__ import print_function
import tensorflow as tf
import numpy as np
from scipy.sparse.linalg import svds

relu = lambda x: 0.5 * (x + abs(x))


def appr_seminmf(M, r):
    """
        Approximate Semi-NMF factorisation.

        Parameters
        ----------
        M: array-like, shape=(n_features, n_samples)
        r: number of components to keep during factorisation
    """

    S, A, B = tf.svd(M, full_matrices=False)
    S = tf.diag(S[0:tf.constant(r - 1)])
    A = tf.matmul(A[:, 0:tf.constant(r - 1)], S)
    B = tf.transpose(B)[0:tf.constant(r - 1), :]

    m, n = M.get_shape().as_list()

    ii = tf.constant(0)
    AA = A[:, ii]
    BB = B[ii, :]
    Atemp = tf.cond(tf.less(tf.reduce_min(B[ii, :]), tf.reduce_min(tf.negative(B[ii, :]))),
                    lambda: tf.reshape(tf.negative(AA), [m, 1]), lambda: tf.reshape(AA, [m, 1]))
    Btemp = tf.cond(tf.less(tf.reduce_min(B[ii, :]), tf.reduce_min(tf.negative(B[ii, :]))),
                    lambda: tf.reshape(tf.negative(BB), [1, n]), lambda: tf.reshape(BB, [1, n]))
    if r > 2:
        for i in range(1, r - 1):
            ii = tf.constant(i)
            AA = tf.reshape(A[:, ii], [m, 1])
            BB = tf.reshape(B[ii, :], [1, n])
            Atemp = tf.cond(tf.less(tf.reduce_min(B[ii, :]), tf.reduce_min(tf.negative(B[ii, :]))),
                            lambda: tf.concat([Atemp, tf.negative(AA)], axis=1), lambda: tf.concat([Atemp, AA], axis=1))
            Btemp = tf.cond(tf.less(tf.reduce_min(B[ii, :]), tf.reduce_min(tf.negative(B[ii, :]))),
                            lambda: tf.concat([Btemp, tf.negative(BB)], axis=0), lambda: tf.concat([Btemp, BB], axis=0))

    if r == 2:
        U = tf.concat([A, tf.negative(A)], axis=1)
    else:
        An = tf.reshape(tf.transpose(tf.negative(tf.reduce_sum(A, 1))), [m, 1])
        U = tf.concat([A, An], 1)

    V = tf.concat([B, tf.zeros((1, n))], 0)

    if r >= 3:
        V = tf.subtract(V, tf.minimum(0.0, tf.reduce_min(B, 0)))
    else:
        V = tf.subtract(V, tf.minimum(0.0, B))

    norm_const = tf.sqrt(tf.cast(tf.multiply(m, n), tf.float32))
    norm = tf.norm(U)

    return tf.multiply(tf.divide(U, norm), norm_const), tf.divide(tf.multiply(V, norm), norm_const)


def init_weights(X, num_components, svd_init=True):
    if svd_init:
        return appr_seminmf(X, num_components)

    m, n = X.get_shape().as_list()
    Z = 0.08 * tf.random_uniform((m, num_components), maxval=1)
    H = 0.08 * tf.random_uniform((num_components, n), maxval=1)

    return Z, H


def dropout(x, p=0):
    if p == 0:
        return x
    else:
        p = 1 - p
        x /= p

        # return x * rng.binomial(x.shape, p=p, dtype=theano.config.floatX)
        return x * tf.contrib.distributions.Binomial(total_count=x.shape, probs=p)


class DSNMF(object):
    def __init__(self, data, layers, pretrain=True, learning_rate=1e-3, iter_num=500, display_step=5):
        """
        Parameters
        ----------
        :param data: array-like, shape=(n_samples, n_features)
        :param layers: list, shape=(n_layers) containing the size of each of the layers
        :param pretrain: pretrain layers using svd
        """

        assert len(layers) > 0, "You have to provide a positive number of layers."

        params = []

        m, n = data.shape
        D = tf.placeholder(shape=(m, n), dtype=tf.float32)
        H = D

        for i, l in enumerate(layers, start=1):
            print('Pretraining {}th layer [{}]\r'.format(i, l))
            Z, H = init_weights(H, l, svd_init=pretrain)
            # params.append(Z)
            params.append(tf.Variable(Z, name='Z_%d' % (i), dtype=tf.float32))

        params.append(tf.Variable(H, name='H_%d' % len(layers), dtype=tf.float32))

        self.params = params
        self.layers = layers

        est = tf.nn.relu(self.params[-1])
        for z in reversed(self.params[0:-1][:]):
            est = tf.nn.relu(tf.matmul(z, est))

        cost = tf.nn.l2_loss(tf.subtract(D, est))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init, feed_dict={D: data})

        for epoch in range(iter_num):
            _, c = sess.run([optimizer, cost], feed_dict={D: data})
            if epoch % display_step == 0:
                print("Epoch {}. Cost [{:.2f}]\r".format(epoch, float(c)))

        self.cost = c
        self.params = [sess.run(p) for p in self.params]

    def get_param_values(self):
        sess = tf.Session()
        return [sess.run(p) for p in self.params]

def main():
    V = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    model = NMF(max_iter=200, learning_rate=0.01, display_step=10, optimizer='mu')
    W, H = model.fit_transform(V, r_components=2, initW=False, givenW=0)
    print(W)
    print(H)
    print(V)
    print(model.inverse_transform(W, H))


if __name__ == '__main__':
    main()