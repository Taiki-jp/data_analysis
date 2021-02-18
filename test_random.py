import random

random.seed(0)

print(random.random())

from my_setting import FindsDir, SetsPath
SetsPath().set()

import tensorflow as tf

tf.random.set_seed(0)
from my_model import MyInceptionAndAttention
m_findsDir = FindsDir("sleep")
o_model = MyInceptionAndAttention(5, 128, 512, m_findsDir)

weights = o_model.model.get_layer('conv2d').get_weights()

type(weights) # >>> list
len(weights) # >>> 2
first_weights = weights[0]
type(first_weights) # >> ndarray
first_weights.shape  # >> (1, 4, 1, 3)
print(first_weights)

"""

first_weights
array([[[[ 0.0981065 ,  0.05300099, -0.23613599]],

        [[ 0.13776547,  0.4503333 , -0.5298845 ]],

        [[-0.02082849, -0.2338962 ,  0.27563977]],

        [[ 0.28503275, -0.2077893 ,  0.06390756]]]], dtype=float32)


[[[[ 0.0981065   0.05300099 -0.23613599]]

  [[ 0.13776547  0.4503333  -0.5298845 ]]

  [[-0.02082849 -0.2338962   0.27563977]]

  [[ 0.28503275 -0.2077893   0.06390756]]]]

"""