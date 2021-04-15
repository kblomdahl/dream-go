# Copyright (c) 2019 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import tensorflow as tf

from .batch_norm import batch_norm_conv2d, relu3
from .dense import dense
from .recompute_grad import recompute_grad


def policy_head(x, mode, params):
    """
    The policy head attached after the residual blocks as described by DeepMind:

    1. A convolution of 8 filters of kernel size 3 × 3 with stride 1
    2. Batch normalisation
    3. A rectifier non-linearity
    4. A fully connected linear layer that outputs a vector of size 19²+1 = 362
       corresponding to logit probabilities for all intersections and the pass
       move
    """
    num_channels = params['num_channels']
    num_samples = params['num_samples']

    def _forward(x, is_recomputing=False):
        """ Returns the result of the forward inference pass on `x` """
        y = batch_norm_conv2d(x, 'conv_1', (3, 3, num_channels, num_samples), mode, params, is_recomputing=is_recomputing)
        y = tf.nn.relu(y)

        y = tf.reshape(y, (-1, 361 * num_samples))
        y = dense(y, 'linear_1', (361 * num_samples, 362), policy_offset_op, mode, params, is_recomputing=is_recomputing)

        return tf.cast(y, tf.float32)

    return recompute_grad(_forward)(x)


def policy_offset_op(shape, dtype=None, partition_info=None):
    """ Initial value for the policy offset, this should roughly correspond to
    the log probability of each move being played. """
    return np.array([
        -7.93991e+00, -6.91853e+00, -6.86255e+00, -6.78094e+00, -6.79361e+00, -6.75976e+00,
        -6.88288e+00, -6.90817e+00, -6.93508e+00, -6.92374e+00, -6.91856e+00, -6.91075e+00,
        -6.87607e+00, -6.75246e+00, -6.79823e+00, -6.80791e+00, -6.86863e+00, -6.89708e+00,
        -7.93729e+00, -6.95779e+00, -6.11830e+00, -5.85974e+00, -5.83566e+00, -5.81966e+00,
        -5.84875e+00, -5.90686e+00, -5.97848e+00, -5.99648e+00, -5.99342e+00, -5.99524e+00,
        -5.96306e+00, -5.88135e+00, -5.83725e+00, -5.81963e+00, -5.84671e+00, -5.85574e+00,
        -6.07402e+00, -6.89741e+00, -6.91472e+00, -5.87616e+00, -5.51456e+00, -5.48398e+00,
        -5.55522e+00, -5.49329e+00, -5.70271e+00, -5.65749e+00, -5.70621e+00, -5.68975e+00,
        -5.69774e+00, -5.66463e+00, -5.68246e+00, -5.43859e+00, -5.59398e+00, -5.44977e+00,
        -5.45890e+00, -5.81432e+00, -6.85663e+00, -6.83055e+00, -5.84429e+00, -5.40160e+00,
        -5.34049e+00, -5.66119e+00, -5.62512e+00, -5.71932e+00, -5.72455e+00, -5.70309e+00,
        -5.69903e+00, -5.70189e+00, -5.71451e+00, -5.68138e+00, -5.59716e+00, -5.64521e+00,
        -5.29867e+00, -5.42794e+00, -5.80074e+00, -6.80807e+00, -6.81930e+00, -5.82896e+00,
        -5.63177e+00, -5.67078e+00, -5.93261e+00, -5.78339e+00, -5.80250e+00, -5.78522e+00,
        -5.79703e+00, -5.79409e+00, -5.79848e+00, -5.78746e+00, -5.77879e+00, -5.76154e+00,
        -5.94899e+00, -5.67992e+00, -5.59753e+00, -5.78787e+00, -6.79474e+00, -6.79318e+00,
        -5.85460e+00, -5.47365e+00, -5.60804e+00, -5.79080e+00, -5.80699e+00, -5.80015e+00,
        -5.81436e+00, -5.81617e+00, -5.80918e+00, -5.81150e+00, -5.80510e+00, -5.77611e+00,
        -5.78804e+00, -5.76476e+00, -5.58303e+00, -5.41241e+00, -5.83056e+00, -6.78050e+00,
        -6.88840e+00, -5.91061e+00, -5.69064e+00, -5.71108e+00, -5.79579e+00, -5.80311e+00,
        -5.81472e+00, -5.81526e+00, -5.81671e+00, -5.81616e+00, -5.81570e+00, -5.80513e+00,
        -5.79622e+00, -5.77254e+00, -5.77513e+00, -5.67571e+00, -5.67228e+00, -5.89279e+00,
        -6.86025e+00, -6.91154e+00, -5.97718e+00, -5.66273e+00, -5.72542e+00, -5.78770e+00,
        -5.81699e+00, -5.81516e+00, -5.81869e+00, -5.81941e+00, -5.81940e+00, -5.81482e+00,
        -5.80754e+00, -5.79365e+00, -5.78832e+00, -5.75882e+00, -5.70202e+00, -5.63253e+00,
        -5.94600e+00, -6.88401e+00, -6.91774e+00, -5.99960e+00, -5.70958e+00, -5.70386e+00,
        -5.80010e+00, -5.81106e+00, -5.81648e+00, -5.81789e+00, -5.81997e+00, -5.81948e+00,
        -5.81279e+00, -5.80583e+00, -5.80135e+00, -5.78998e+00, -5.77203e+00, -5.68193e+00,
        -5.67815e+00, -5.96948e+00, -6.88898e+00, -6.91699e+00, -5.99684e+00, -5.69323e+00,
        -5.68440e+00, -5.79516e+00, -5.81060e+00, -5.81611e+00, -5.81406e+00, -5.81620e+00,
        -5.80901e+00, -5.81298e+00, -5.80653e+00, -5.79696e+00, -5.78196e+00, -5.76473e+00,
        -5.65428e+00, -5.66398e+00, -5.96876e+00, -6.89641e+00, -6.92151e+00, -5.99694e+00,
        -5.71110e+00, -5.71325e+00, -5.79821e+00, -5.80778e+00, -5.81212e+00, -5.81205e+00,
        -5.81020e+00, -5.81116e+00, -5.80801e+00, -5.79830e+00, -5.79276e+00, -5.78653e+00,
        -5.77101e+00, -5.68899e+00, -5.69274e+00, -5.97098e+00, -6.90131e+00, -6.89817e+00,
        -5.95772e+00, -5.64660e+00, -5.72654e+00, -5.77678e+00, -5.80212e+00, -5.80607e+00,
        -5.80127e+00, -5.80551e+00, -5.80743e+00, -5.80042e+00, -5.79346e+00, -5.79025e+00,
        -5.78733e+00, -5.75338e+00, -5.69506e+00, -5.63437e+00, -5.95747e+00, -6.88818e+00,
        -6.86408e+00, -5.86964e+00, -5.67686e+00, -5.70769e+00, -5.79369e+00, -5.78719e+00,
        -5.79913e+00, -5.80025e+00, -5.80054e+00, -5.80132e+00, -5.79529e+00, -5.78667e+00,
        -5.78821e+00, -5.76922e+00, -5.76675e+00, -5.69570e+00, -5.68074e+00, -5.90285e+00,
        -6.86338e+00, -6.76061e+00, -5.80263e+00, -5.41706e+00, -5.58843e+00, -5.78328e+00,
        -5.79366e+00, -5.78934e+00, -5.79841e+00, -5.79591e+00, -5.79041e+00, -5.79060e+00,
        -5.78705e+00, -5.78000e+00, -5.77674e+00, -5.75681e+00, -5.57623e+00, -5.50113e+00,
        -5.85626e+00, -6.78012e+00, -6.79139e+00, -5.80594e+00, -5.58041e+00, -5.65286e+00,
        -5.94338e+00, -5.77647e+00, -5.78968e+00, -5.77167e+00, -5.78232e+00, -5.76841e+00,
        -5.77241e+00, -5.75895e+00, -5.78530e+00, -5.76951e+00, -5.88238e+00, -5.64461e+00,
        -5.61617e+00, -5.82903e+00, -6.80791e+00, -6.81286e+00, -5.84175e+00, -5.48596e+00,
        -5.28293e+00, -5.71807e+00, -5.60505e+00, -5.71724e+00, -5.70963e+00, -5.68757e+00,
        -5.65039e+00, -5.67046e+00, -5.68983e+00, -5.69079e+00, -5.58636e+00, -5.60082e+00,
        -5.39104e+00, -5.38788e+00, -5.85818e+00, -6.81584e+00, -6.83461e+00, -5.85197e+00,
        -5.47331e+00, -5.40193e+00, -5.63715e+00, -5.47135e+00, -5.68295e+00, -5.64977e+00,
        -5.67997e+00, -5.64680e+00, -5.67367e+00, -5.61327e+00, -5.67216e+00, -5.50078e+00,
        -5.53072e+00, -5.40751e+00, -5.52960e+00, -5.87713e+00, -6.89602e+00, -6.89446e+00,
        -6.07997e+00, -5.83860e+00, -5.78284e+00, -5.77460e+00, -5.81606e+00, -5.88522e+00,
        -5.95163e+00, -5.97232e+00, -5.95954e+00, -5.96527e+00, -5.94048e+00, -5.88465e+00,
        -5.82810e+00, -5.82003e+00, -5.84255e+00, -5.88531e+00, -6.11968e+00, -6.92480e+00,
        -7.88397e+00, -6.89418e+00, -6.83908e+00, -6.78821e+00, -6.75784e+00, -6.75053e+00,
        -6.85545e+00, -6.88249e+00, -6.88945e+00, -6.88525e+00, -6.88876e+00, -6.86828e+00,
        -6.83631e+00, -6.75981e+00, -6.76317e+00, -6.74771e+00, -6.86408e+00, -6.90874e+00,
        -7.91371e+00, -6.27113e+00
    ])
