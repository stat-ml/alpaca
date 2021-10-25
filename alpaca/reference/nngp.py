import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse
import time
import sys
import time
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split


def simple_cov(_x, _y):
    return np.mean((_x-np.mean(_x))*(_y-np.mean(_y)), axis = 1)


def simple_covs(a, b):
    ac = a - a.mean(axis=-1, keepdims=True)
    bc = (b - b.mean(axis=-1, keepdims=True)) / b.shape[-1]
    return np.dot(ac, bc.T).T


def greedy_batch_acquizition(values, sample_size):
    return np.argsort(values)[::-1][:sample_size]


def update_learning_sets(X_train,
                         y_train,
                         X_pool,
                         y_pool,
                         ue_values,
                         acquizition = greedy_batch_acquizition,
                         sample_size = 10):
    inds = greedy_batch_acquizition(ue_values, sample_size)
    X_train = np.concatenate([X_train, X_pool[inds]])
    y_train = np.concatenate([y_train, y_pool[inds]])
    X_pool = np.delete(X_pool, inds, axis=0)
    y_pool = np.delete(y_pool, inds, axis=0)
    
    return X_train, y_train, X_pool, y_pool


def iterate_minibatches(inputs, targets, batchsize, shuffle=True):
    """
    Produces batches
    """
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    if batchsize > len(inputs):
        yield inputs[indices], targets[indices]
    else:
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]
        if len(inputs) % batchsize > 0:
            if shuffle:
                excerpt = indices[start_idx+batchsize:len(inputs)]
            else:
                excerpt = slice(start_idx+batchsize, len(inputs))
            yield inputs[excerpt], targets[excerpt]

            
class NN:
    def __init__(
        self, ndim=1, layers=None, initializer=None, activation=None,
        random_state=42, dropout_layers=None
    ):
        if layers is None:
            layers = [32, 16]
        if initializer is None:
            initializer = tf.contrib.layers.xavier_initializer()
        if activation is None:
            activation = tf.nn.leaky_relu

        # for arg, value in locals().items():  # oh my god, what have i done
        #     if arg == 'self':
        #         continue
        #     self.__dict__[arg] = value

        if dropout_layers is None:
            dropout_layers = [True] * (len(layers)-1)

        self.input_data = tf.placeholder(tf.float32, [None, ndim])
        self.answer = tf.placeholder(tf.float32, [None, 1])
        self.keep_probability_ = tf.placeholder(tf.float32, name='keep_probability')
        self.keep_probability_inner_ = tf.placeholder(tf.float32, name='keep_probability')
        self.l2_reg_ = tf.placeholder(tf.float32, name='l2reg')
        
        W1 = tf.get_variable(name='W0', shape=(ndim, layers[0]), initializer=initializer)
        b1 = tf.get_variable(name='b0', shape=(layers[0],), initializer=initializer)

        h1 = activation(tf.matmul(self.input_data, W1) + b1)
        
        h_drop1 = tf.nn.dropout(h1, self.keep_probability_, noise_shape=[1, layers[0]])
        
        self.Ws = [W1]
        self.bs = [b1]
        self.hs = [h_drop1]
        
        for cnt_layer in range(1, len(layers)):
            self.Ws.append(
                tf.get_variable(
                    name = f'W{cnt_layer}',
                    shape = (layers[cnt_layer - 1], layers[cnt_layer]),
                    initializer = initializer))
            self.bs.append(
                tf.get_variable(
                    name = f'b{cnt_layer}',
                    shape = (layers[cnt_layer],),
                    initializer = initializer))
            h = activation(tf.matmul(self.hs[-1], self.Ws[-1]) + self.bs[-1])
            h = tf.nn.dropout(h,
                    self.keep_probability_ if dropout_layers[cnt_layer-1]
                                           else self.keep_probability_inner_,
                    noise_shape = [1, layers[cnt_layer]])
            self.hs.append(h)
                
        self.Ws.append(
            tf.get_variable(
                    name=f'W{len(layers)}',
                    shape=(layers[-1], 1),
                    initializer=initializer))
        self.bs.append(
            tf.get_variable(
                    name=f'b{len(layers)}',
                    shape=(1,),
                    initializer=initializer))
        self.output = activation(tf.matmul(self.hs[-1], self.Ws[-1]) + self.bs[-1])
        
        self.l2_regularizer = sum(tf.nn.l2_loss(Wxxx) for Wxxx in self.Ws) 
        self.mse = tf.losses.mean_squared_error(predictions = self.output, 
                                           labels = self.answer)
        self.loss = self.mse + self.l2_reg_*self.l2_regularizer
        self.train_step = tf.train.AdamOptimizer(learning_rate=1e-4).\
                                minimize(self.loss)
        
        # misc part for nngp eval
        self.KK_ = tf.placeholder(tf.float32)
        self.Q_ = tf.placeholder(tf.float32,
                                 [None, None])
        self.K_train_cov_inv_ = tf.placeholder(tf.float32, [None, None])
        self.gpue_tf = \
            tf.linalg.tensor_diag_part(self.KK_ - \
                                       tf.matmul(
                                           tf.matmul(
                                               tf.transpose(self.Q_), 
                                               self.K_train_cov_inv_), 
                                               self.Q_))
        
    def train(
        self, session, X_train, y_train, X_test, y_test, X_val, y_val,
        epochs=10000, early_stopping=True, validation_window=100,
        patience=3, keep_prob=1., l2_reg=0, verbose=True, batch_size=500,
    ):
        
        previous_error = 1e+10
        current_patience = patience
        
        for epoch_num in range(1, 1 + epochs):
            for X_batch, y_batch in iterate_minibatches(X_train, y_train, batch_size):
                session.run(self.train_step,
                            feed_dict={self.input_data: X_batch, 
                                       self.answer: y_batch, 
                                       self.keep_probability_inner_: keep_prob,
                                       self.keep_probability_: keep_prob, 
                                       self.l2_reg_: l2_reg})

            if (early_stopping) and epoch_num % validation_window == 0:
                rmse_train = np.sqrt(
                    session.run(self.mse, 
                             feed_dict={self.input_data: X_train, 
                                        self.answer: y_train, 
                                        self.keep_probability_inner_: 1.,
                                        self.keep_probability_: 1.}))
                rmse_test = np.sqrt(
                    session.run(self.mse, 
                             feed_dict={self.input_data: X_test, 
                                        self.answer: y_test, 
                                        self.keep_probability_inner_: 1.,
                                        self.keep_probability_: 1.}))       
                rmse_val = np.sqrt(
                    session.run(self.mse, 
                             feed_dict={self.input_data: X_val, 
                                        self.answer: y_val, 
                                        self.keep_probability_inner_: 1.,
                                        self.keep_probability_: 1.}))

                if rmse_val > previous_error:
                    current_patience -= 1
                else:
                    previous_error = rmse_val
                    current_patience = patience
                if verbose:
                    print(f'[{epoch_num}]'+\
                          f' RMSE train:{rmse_train:.3f}'+\
                          f' test:{rmse_test:.3f}'+\
                          f' val:{rmse_val:.3f}'+\
                          f' patience:{current_patience}')
                if current_patience <= 0:
                    if verbose:
                        print(f'No patience left at epoch {epoch_num}.'+\
                        ' Early stopping.')
                    break
        return epoch_num, rmse_train, rmse_test, rmse_val

    def predict(self, session, data, probability=1., probabitily_inner=1.):
        return session.run(self.output,
                           feed_dict={self.input_data: data,
                                      self.keep_probability_inner_: probabitily_inner,
                                      self.keep_probability_: probability})

    def get_mcd_UE(self, session, X_pool, nn_runs=25, probability=.5, use_inner=False):
        # nn inference
        mcd_realizations = np.zeros((X_pool.shape[0], nn_runs))
        for nn_run in range(nn_runs):
            mcd_realizations[:, nn_run] = \
                np.ravel(self.predict(session,
                                      data = X_pool,
                                      probability = probability,
                                      probabitily_inner = probability if use_inner
                                                          else 1.))
        return np.ravel(np.std(mcd_realizations, axis = 1))
    
    def get_nngp_UE(
        self, session, X_train, y_train, X_pool, y_pool, random_subsampling = -1,
        nn_runs=25, diag_eps=1e-6, inference_batch_size=1000, probability=.5,
        use_inner=False
    ):
        # data preparation
        train_len = len(X_train)
        if random_subsampling > 0:
            train_len = min(random_subsampling, train_len)
            random_train_inds = \
                np.random.permutation(range(train_len))[:random_subsampling]
            train_pool_samples = \
                np.concatenate([X_train[random_train_inds],
                                X_pool])
        else:
            train_pool_samples = \
                np.concatenate([X_train,
                                X_pool])
        # nn inference
        mcd_realizations = np.zeros((train_pool_samples.shape[0], nn_runs))
        for nn_run in range(nn_runs):
            mcd_realizations[:, nn_run] = \
                np.ravel(self.predict(session,
                                      data=train_pool_samples,
                                      probability=probability,
                                      probabitily_inner=probability if use_inner
                                                          else 1.))
        cov_matrix_train = np.cov(mcd_realizations[:train_len, :], 
                                  ddof = 0)
        cov_matrix_inv = np.linalg.inv(cov_matrix_train\
                                       + np.eye(train_len)*diag_eps)
        gp_ue = np.zeros((len(X_pool), ))
        
        cnt = 0 # for the case of inference_batch_size > len(X_pool)
        for cnt in range(len(X_pool) // inference_batch_size):
            left_ind = train_len + cnt*inference_batch_size
            right_ind = train_len + cnt*inference_batch_size+inference_batch_size
            pool_samples = mcd_realizations[left_ind:right_ind,:]
            Qs = simple_covs(mcd_realizations[:train_len,:], pool_samples).T
            KKs = np.var(pool_samples, axis=1)
            ws = session.run(self.gpue_tf, {
                             self.Q_: Qs,
                             self.K_train_cov_inv_: cov_matrix_inv,
                             self.KK_: KKs})
            gp_ue_currents = [0 if w < 0 else np.sqrt(w) for w in np.ravel(ws)]
            gp_ue[(left_ind - train_len):(right_ind - train_len)] = gp_ue_currents
        right_ind = train_len + cnt*inference_batch_size+inference_batch_size
        pool_samples = mcd_realizations[right_ind:,:]
        Qs = simple_covs(mcd_realizations[:train_len,:], pool_samples).T
        KKs = np.var(pool_samples, axis = 1)
        ws = session.run(self.gpue_tf, {
                             self.Q_: Qs,
                             self.K_train_cov_inv_: cov_matrix_inv,
                             self.KK_: KKs})
        gp_ue_currents = [0 if w < 0 else np.sqrt(w) for w in np.ravel(ws)]
        gp_ue[right_ind:] = gp_ue_currents
        return np.ravel(gp_ue)
        
