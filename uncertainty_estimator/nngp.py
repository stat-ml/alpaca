import tensorflow as tf
import numpy as np


class NNGP:
    """
    Estimate uncertainty for samples with MCDUE approach
    """
    def __init__(
        self, net, random_subsampling=-1, nn_runs=25, diag_eps=1e-6,
        inference_batch_size=1000, probability=.5, use_inner=False
    ):
        self.net = net
        self.random_subsampling = random_subsampling
        self.nn_runs = nn_runs
        self.diag_eps = diag_eps
        self.batch_size = inference_batch_size
        self.probability = probability
        self.use_inner = use_inner

        self.gpue_tf = self._build_ue_computation_graph()

    def estimate(self, X_pool, X_train, y_train):
        # data preparation
        train_pool_samples, train_len = self._generate_samples(X_train, X_pool)

        # monte-carlo dropout nn inference
        mcd_predictions = self._mcd_predict(train_pool_samples)

        # covariance matrix with regularization
        cov_matrix_train = np.cov(mcd_predictions[:train_len, :], ddof=0)
        cov_matrix_inv = np.linalg.inv(cov_matrix_train + np.eye(train_len)*self.diag_eps)

        gp_ue = np.zeros((len(X_pool), ))
        for i in range(train_len, train_len+len(X_pool), self.batch_size):
            left = i
            right = min(i+self.batch_size, len(mcd_predictions))

            pool_samples = mcd_predictions[left:right, :]
            Qs = self.simple_covs(mcd_predictions[:train_len, :], pool_samples).T
            KKs = np.var(pool_samples, axis=1)

            feed_dict = {
                 self.Q_: Qs,
                 self.K_train_cov_inv_: cov_matrix_inv,
                 self.KK_: KKs
            }
            with tf.Session() as sess:
                ws = sess.run(self.gpue_tf, feed_dict)

            gp_ue_currents = [0 if w < 0 else np.sqrt(w) for w in np.ravel(ws)]
            gp_ue[(left - train_len):(right - train_len)] = gp_ue_currents
            
        return np.ravel(gp_ue)

    def _build_ue_computation_graph(self):
        self.KK_ = tf.placeholder(tf.float32)
        self.Q_ = tf.placeholder(tf.float32, [None, None])
        self.K_train_cov_inv_ = tf.placeholder(tf.float32, [None, None])
        qt_K_q = tf.matmul(tf.matmul(tf.transpose(self.Q_), self.K_train_cov_inv_), self.Q_)
        return tf.linalg.tensor_diag_part(self.KK_ - qt_K_q)

    def _generate_samples(self, X_train, X_pool):
        train_len = len(X_train)
        if self.random_subsampling > 0:
            train_len = min(self.random_subsampling, train_len)
            random_train_inds = np.random.permutation(range(train_len))[:self.random_subsampling]
            train_pool_samples = np.concatenate([X_train[random_train_inds], X_pool])
        else:
            train_pool_samples = np.concatenate([X_train, X_pool])

        return train_pool_samples, train_len

    def _mcd_predict(self, train_pool_samples):
        mcd_predictions = np.zeros((train_pool_samples.shape[0], self.nn_runs))
        for nn_run in range(self.nn_runs):
            prediction = self._net_predict(train_pool_samples)
            mcd_predictions[:, nn_run] = np.ravel(prediction)
        return mcd_predictions

    def _net_predict(self, train_pool_samples):
        probability_inner = self.probability if self.use_inner else 1.
        
        return self.net.predict(
            data=train_pool_samples, probability=self.probability,
            probabitily_inner=probability_inner
        )
    
    @staticmethod
    def simple_covs(a, b):
        ac = a - a.mean(axis=-1, keepdims=True)
        bc = (b - b.mean(axis=-1, keepdims=True)) / b.shape[-1]
        return np.dot(ac, bc.T).T

