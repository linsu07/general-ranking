import six
import tensorflow as tf
from tensorflow.python.layers.core import dense
from tensorflow.contrib.layers import xavier_initializer
'''
  * Created by linsu on 2019/1/25.
  * mailto: lsishere2002@hotmail.com
'''

class MLPLayer(tf.layers.Layer):
    def __init__(self,params,is_trainning=False, dtype=tf.float32, name="mlp"):
        super(MLPLayer, self).__init__(is_trainning, name, dtype)
        self.params = params

    def build(self, _):
        self.built = True


    def call(self, features_list, **kwargs):
        [context_features,per_examplefeatures] =features_list
        is_training = self.trainable
        features= []
        for _,v in six.iteritems(context_features):
            features.append(v)
        for name ,v in six.iteritems(per_examplefeatures):
            if name =="user_item_list_features":
                va = tf.squeeze(v,1)
            else:
                va = v
            features.append(va)
        features = tf.concat(features,-1)
        cur_layer = tf.layers.batch_normalization(features, training=is_training)
        for i, layer_width in enumerate(int(d) for d in self.params["hidden_layer_dims"]):
            cur_layer = dense(cur_layer, units=layer_width,kernel_initializer = xavier_initializer())
            cur_layer = tf.layers.batch_normalization(cur_layer, training=is_training)
            cur_layer = tf.nn.relu(cur_layer)
            tf.summary.scalar("fully_connected_{}_sparsity".format(i),
                              tf.nn.zero_fraction(cur_layer))
        cur_layer = tf.layers.dropout(
            cur_layer, rate=self.params["drop_out_rate"], training=is_training)
        logits = tf.layers.dense(cur_layer, units=1)
        return logits
