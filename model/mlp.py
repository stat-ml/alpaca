import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, layer_sizes):
        super(MLP, self).__init__()

        self.fcs = []
        for i, layer in enumerate(layer_sizes[:-1]):
            self.fc = nn.Linear(layer, layer_sizes[i+1])  # we use self to register as torch parameter
            self.fcs.append(self.fc)
        self.relu = nn.ReLU()

    def forward(self, x, dropout=True):
        out = x
        for fc in self.fcs:
            out = self.relu(fc(out))
        return out

    def fit(self, train_loader, val_loader, learning_rate=0.001, num_epochs=10):
        print('start fitting')
        device = 'gpu' if torch.cuda.is_available() else 'cpu'
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # Train the model
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                # Move tensors to the configured device
                images = images.reshape(-1, 28 * 28).to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = self(images, False)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % 100 == 0:
                    self._print_status(epoch, num_epochs, i, loss.item())

    def _print_status(self, epoch, num_epochs, i, loss):
        print('Epoch [{}/{}], Step [{}], Loss: {:.4f}'
              .format(epoch + 1, num_epochs, i + 1, loss))

# class MLP:
#     """
#     Simple fully-connected network
#     It can make dropout on inference (that's helps with MCDUE and NNGP)
#     """
#     def __init__(
#             self, ndim, layers, initializer=None, activation=None,
#             dropout_layers=None):
#
#         self._build_net(layers, dropout_layers, initializer, activation, ndim)
#
#     def set_session(self, session):
#         self.session = session
#
#     def _build_net(self, layers, dropout_layers, initializer, activation, ndim):
#         if initializer is None:
#             initializer = tf.keras.initializers.glorot_normal()
#         if activation is None:
#             activation = tf.nn.leaky_relu
#         if dropout_layers is None:
#             dropout_layers = [True] * (len(layers)-1)
#
#         self.input_data = tf.placeholder(tf.float32, [None, ndim])
#         self.answer = tf.placeholder(tf.float32, [None, 1])
#         self.keep_probability_ = tf.placeholder(tf.float32, name='keep_probability')
#         self.keep_probability_inner_ = tf.placeholder(tf.float32, name='keep_probability')
#         self.l2_reg_ = tf.placeholder(tf.float32, name='l2reg')
#
#         # First layer
#         W0 = tf.get_variable(name='W0', shape=(ndim, layers[0]), initializer=initializer)
#         b0 = tf.get_variable(name='b0', shape=(layers[0],), initializer=initializer)
#         h0 = activation(tf.matmul(self.input_data, W0) + b0)
#         h_drop0 = tf.nn.dropout(h0, self.keep_probability_, noise_shape=[1, layers[0]])
#
#         self.Ws = [W0]
#         self.bs = [b0]
#         self.hs = [h_drop0]
#
#         # Inner layers
#         for i in range(1, len(layers)):
#             shape = (layers[i - 1], layers[i])
#             Wi = tf.get_variable(name=f'W{i}', shape=shape, initializer=initializer)
#             self.Ws.append(Wi)
#
#             bi = tf.get_variable(name=f'b{i}', shape=(layers[i],), initializer=initializer)
#             self.bs.append(bi)
#
#             hi = activation(tf.matmul(self.hs[-1], Wi) + bi)
#
#             if dropout_layers[i-1]:
#                 dropout_probability = self.keep_probability_
#             else:
#                 dropout_probability = self.keep_probability_inner_
#
#             h = tf.nn.dropout(hi, dropout_probability, noise_shape=[1, layers[i]])
#             self.hs.append(h)
#
#         # Last layer
#         Wn = tf.get_variable(name=f'W{len(layers)}', shape=(layers[-1], 1), initializer=initializer)
#         self.Ws.append(Wn)
#         bn = tf.get_variable(name=f'b{len(layers)}', shape=(1,), initializer=initializer)
#         self.bs.append(bn)
#         self.output = activation(tf.matmul(self.hs[-1], self.Ws[-1]) + self.bs[-1])
#
#         # Optimization and metrics
#         self.l2_regularizer = sum(tf.nn.l2_loss(Wxxx) for Wxxx in self.Ws)
#         self.mse = tf.losses.mean_squared_error(predictions=self.output, labels=self.answer)
#         self.loss = self.mse + self.l2_reg_*self.l2_regularizer
#         optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
#         self.train_step = optimizer.minimize(self.loss)
#
#     def train(
#         self, X_train, y_train, X_val, y_val,
#         epochs=10000, early_stopping=True, validation_window=100,
#         patience=3, keep_prob=1., l2_reg=0, verbose=True, batch_size=500,
#     ):
#         previous_error = 1e+10
#         current_patience = patience
#
#         for epoch_num in range(1, 1 + epochs):
#             for X_batch, y_batch in self.iterate_minibatches(X_train, y_train, batch_size):
#                 feed_dict = {
#                     self.input_data: X_batch,
#                     self.answer: y_batch,
#                     self.keep_probability_inner_: keep_prob,
#                     self.keep_probability_: keep_prob,
#                     self.l2_reg_: l2_reg
#                 }
#                 self.session.run(self.train_step, feed_dict=feed_dict)
#
#             if early_stopping and epoch_num % validation_window == 0:
#                 rmse_train = self._rmse(X_train, y_train)
#                 rmse_val = self._rmse(X_val, y_val)
#
#                 if rmse_val > previous_error:
#                     current_patience -= 1
#                 else:
#                     previous_error = rmse_val
#                     current_patience = patience
#                 if verbose:
#                     self._print_rmse(epoch_num, rmse_train, rmse_val, current_patience)
#                 if current_patience <= 0:
#                     if verbose:
#                         self._print_no_patience(epoch_num)
#                     break
#         return epoch_num, rmse_train, rmse_val
#
#     def predict(self, data, probability=1., probabitily_inner=1.):
#         feed_dict = {
#             self.input_data: data,
#             self.keep_probability_inner_: probabitily_inner,
#             self.keep_probability_: probability
#         }
#         return self.session.run(self.output, feed_dict=feed_dict)
#
#     @staticmethod
#     def iterate_minibatches(inputs, targets, batch_size, shuffle=False):
#         assert len(inputs) == len(targets)
#
#         indices = np.arange(len(inputs))
#         if shuffle:
#             np.random.shuffle(indices)
#
#         for left in range(0, len(inputs), batch_size):
#             right = min(left + batch_size, len(inputs))
#             batch_indices = indices[left:right]
#             yield inputs[batch_indices], targets[batch_indices]
#
#     def _rmse(self, X, y):
#         feed_dict = {
#             self.input_data: X,
#             self.answer: y,
#             self.keep_probability_inner_: 1.,
#             self.keep_probability_: 1.
#         }
#         return np.sqrt(self.session.run(self.mse, feed_dict=feed_dict))
#
#     @staticmethod
#     def _print_rmse(epoch, rmse_train, rmse_val, patience):
#         print(
#             f'[{epoch}]',
#             f' RMSE train:{rmse_train:.3f}',
#             f' val:{rmse_val:.3f}',
#             f' patience:{patience}'
#         )
#
#     @staticmethod
#     def _print_no_patience(epoch):
#         print(f'No patience left at epoch {epoch}. Early stopping.')

