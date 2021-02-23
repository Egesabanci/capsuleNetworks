"""Capsule Networks basic implementation according to the paper:
Dynamic Routing Between Capsules - Sara Sabour, Nicholas Frosst, Geoffrey E Hinton
Paper Link: https://arxiv.org/abs/1710.09829"""


import numpy as np
import tensorflow as tf

# Squash Function
def squash(vector, ax = -1, epsilon = 1e-7):

    """
    Arguments:
        - vector    : vector for squashing
        - ax        : axis for tf.reduce_sum() function
        - epsilon   : for create safe norm
    """
    
    s_squared = tf.reduce_sum(vector, axis = ax, keepdims = True)
    # safe norm for not to divide by absolute zero
    safe_norm = tf.sqrt(s_squared + tf.keras.backend.epsilon(),
                        name = "square_root_squash_safe_norm")
    
    squash_factor = s_squared / (1.0 + s_squared)
    main_vector = vector / safe_norm

    return squash_factor * main_vector

# Primary Capsules
def PrimaryCapsule(inputs, capsule_dim, capsule_maps, 
                   conv_shape, kernel, padding, stride):

    # output number of capsule
    n_caps = capsule_maps * (conv_shape ** 2)
    
    # second convolution layer after initial convolution
    conv_primary1 = tf.keras.layers.Conv2D(filters = capsule_dim * capsule_maps,
                                          kernel_size = kernel,
                                          strides = stride,
                                          padding = padding,
                                          activation = tf.nn.sigmoid,
                                          name = "Conv2_Primary1")(inputs)
    
    # reshape the vector
    reshaped_primary = tf.reshape(conv_primary1, shape = [-1, n_caps, capsule_dim])

    # squash reshaped vector
    sq_vector_primary = tf.keras.layers.Lambda(squash)(reshaped_primary)

    return sq_vector_primary

# Capsule Layers - Routing algorithm works here
class CapsuleLayers(object):

    def __init__(self, input_n_cap, 
                 input_n_dims, num_capsules, capsule_dims, 
                 kernel_initializer = "glorot_uniform", 
                 bias_initializer = "zeros"):
        
        """
        Arguments:
            - input_n_cap   : primary layers, number of capsules
            - input_n_dims  : primary layers, number of dimensions
            - num_capsules  : secondary number of capsules
            - capsule_dims  : secondary capsule dimensions
        """
        
        self.num_capsules = num_capsules    # secondary layer num capsules
        self.capsule_dims = capsule_dims    # secondary num capsule dimensions
        self.inp_n_caps = input_n_cap   # primary num capsules
        self.inp_n_dims = input_n_dims  # primary num capsule dimensions

        self.kernel_initializer = kernel_initializer    # w matrix initializer
        self.bias_initializer = bias_initializer        # coupling coefficients
    
    def build_matrix(self):
        
        # transform matrix - weights
        w_init = tf.random.normal(
                 shape = [self.inp_n_caps, self.num_capsules,
                          self.capsule_dims, self.inp_n_dims],
                 dtype = tf.float32, stddev = 0.1, name = "weight_matrix")
        
        w = tf.Variable(w_init, name = "w_matrix")
        
        # tiled weight matrix
        self.w_tiled = tf.tile(w, [1, 1, 1, 1], name = "tiled_weight_matrix")
        
        return self.w_tiled

    # routing layer - routing iteration = 2
    def routing(self, primary_output, weight_matrix):

        # expand primary output
        primary_expanded = tf.expand_dims(tf.expand_dims(primary_output, -1), 2)

        # tile primary expanded tensor
        primary_tiled = tf.tile(primary_expanded, [1, 1, self.num_capsules, 1, 1],
                                name = "tiled_primary")

        # secondary capsule predicted
        caps_2_predicted = tf.matmul(weight_matrix, primary_tiled,
                                     name = "predicted_caps_2")

        # raw weights to train
        raw_w1 = tf.zeros(
            shape = [self.inp_n_caps, self.num_capsules, 1, 1],
            dtype = tf.float32, name = "raw_weights_2")
        
        # softmax to raw_weights
        soft_raw_weights1 = tf.nn.softmax(raw_w1, name = "softmax_weights")

        # predictions and sum for squashing
        weighted_preds = tf.multiply(soft_raw_weights1, caps_2_predicted, 
                                     name = "weighted_preds_round1")
        weighted_preds_sum = tf.reduce_sum(weighted_preds, axis = 1, keepdims = True)

        # caps 2 output - first iteration
        caps2_round1 = squash(weighted_preds_sum, ax = -2)

        # agreement calculation
        caps2_round1_tiled = tf.tile(caps2_round1, [1, self.inp_n_caps, 1, 1, 1],
                                     name = "tiled_caps_1_round_1")
        agreement = tf.matmul(caps2_round1, caps2_round1_tiled, transpose_a = True,
                              name = "agreement")
        
        # update routing weights
        raw_w2 = tf.add(raw_w1, agreement, name = "raw_weights_2")

        # weighted prediction and sum - round 2
        softmax_raw_weight2 = tf.nn.softmax(raw_w2, name = "softmax_raw_weights2")
        weighted_preds2 = tf.matmul(caps_2_predicted, softmax_raw_weight2,
                                    name = "weighted_preds_round2")
        weighted_preds_sum2 = tf.reduce_sum(weighted_preds2, axis = 1, keepdims = True)

        caps2_round2 = squash(weighted_preds_sum2, ax = -2)

        # reshape to target vector
        reshaped_out = tf.squeeze(caps2_round2, axis = [1, -1],
                                  name = "squeezed_reshaped_output")

        return reshaped_out

# Lenght calculations
class Length(object):

    def calc_len(self, vector, epsilon = tf.keras.backend.epsilon(), 
                 name = None, keep_dims = True):
        
        """ 
        Arguments:
            - vector        : input tensor
            - ax            : calculating axis
            - epsilon       : the additional value to the
                              equation norm. This value's purpose
                              is, to avoid the error of 'divide by zero'.
            - name          : name of the proccess
            - keep_dims     : dimension control during calculations
        """
        
        self.vector = vector  
        self.epsilon = epsilon
        self.name = name
        self.keep_dims = keep_dims

        # add epsilon to the equation to avoid 'divide by zero' uncertainity
        squared_safe_norm =  tf.reduce_sum(tf.square(self.vector), axis = -1,
                                                     keepdims = self.keep_dims,
                                                    name = "epsilon_added_safe_norm")

        # square root of safe norm to minimize the effects of epsilon
        square_root_of_safe_norm = tf.sqrt(squared_safe_norm, name = "final_safe_norm")

        return square_root_of_safe_norm

# Masking for decoder
class Mask(object):

    def masked(self, x_inputs, y_labels):

        """
        Arguments:
            - x_inputs  : images to be masked
            - y_labels  : true labels to mask images
        """
    
        self.x_inputs = x_inputs    # pred inputs
        self.y_labels = y_labels    # true labels

        x = (self.x_inputs - tf.keras.backend.max(self.x_inputs, 1, True)) / \
             tf.keras.backend.epsilon() + 1
        mask = tf.keras.backend.clip(x, 0, 1)

        input_masked = tf.keras.backend.batch_dot(self.x_inputs, mask, [1, 1])

        return input_masked

# Decoder for reconstruction of images - 2 Dense layers
def Decoder(masked_input, target_shape, dense_units,
            output_units, activation = "relu"):

    # parameter control
    assert len(dense_units) == 2
    
    # 3 fully connected layer to reconstruct
    dense1 = tf.keras.layers.Dense(dense_units[0], activation, 
                                   use_bias = True, name = "fc1")(masked_input)

    dense2 = tf.keras.layers.Dense(dense_units[1], activation, 
                                   name = "fc2")(dense1)
    
    output_dense = tf.keras.layers.Dense(output_units, tf.nn.softmax, 
                                         name = "output_fc3")(dense2)

    # reshape output to get target shape of the image after reconstruction
    reshaped_image = tf.keras.layers.Reshape(target_shape)(output_dense)

    return reshaped_image

# Margin Loss from paper
def margin_loss(y_true, y_pred):

    # margin loss equation from the paper - (dynamic routing between capsules)
    L = y_true * tf.keras.backend.square(tf.keras.backend.maximum(0., 0.9 - y_pred)) \
        + 0.5 * (1 - y_true) * tf.keras.backend.square(tf.keras.backend.maximum(0., y_pred - 0.1))

    # mean margin loss
    mean_m_loss = tf.keras.backend.mean(tf.keras.backend.sum(L, 1))

    return mean_m_loss