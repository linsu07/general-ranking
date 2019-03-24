import six
import tensorflow as tf
import os
from tensorflow.python.estimator.export.export import build_raw_serving_input_receiver_fn, \
    build_parsing_serving_input_receiver_fn, ServingInputReceiver
from tensorflow.python.lib.io import file_io
'''
  * Created by linsu on 2019/1/22.
  * mailto: lsishere2002@hotmail.com
'''
class SparkInput():
    def __init__(self, params):
        self.params = params
        self.conttext_spec = {
            # self.params["user_car_serial"]:tf.VarLenFeature(tf.string),
            # self.params["item_hot_score"]:tf.VarLenFeature(tf.float32),
            self.params["item_relevance"]:tf.VarLenFeature(tf.float32),
            self.params["history_item_relevance"]:tf.VarLenFeature(tf.float32)
        }
        self.sequence_spec = {
            self.params["user_history_item_title"]:tf.VarLenFeature(tf.string),
            self.params["item_title"]:tf.VarLenFeature(tf.string)
        }
        # self.serving_spec = dict()
        # for feature,v in six.iteritems(self.sequence_spec):
        #     self.serving_spec[feature] = v
        for feature_name in self.params["item_feature_list"]:
            self.conttext_spec[feature_name] = tf.VarLenFeature(tf.string)
            # self.serving_spec[feature_name] = tf.VarLenFeature(tf.string)

    def get_data_dir(self, mode: tf.estimator.ModeKeys, params):
        return os.path.join(params["data_dir"], "train") \
            if mode == tf.estimator.ModeKeys.TRAIN else os.path.join(params["data_dir"], "evaluation")

    def input_fn(self, mode: tf.estimator.ModeKeys, params, data_dir):
        file_paths = file_io.get_matching_files(os.path.join(data_dir, "part-r-*"))#返回一个列表
        data_set = tf.data.TFRecordDataset(file_paths, buffer_size=800 * 1024)
        batch_size = params["batch_size"]
        def parse(raw):
            context_dic,seq_dic =\
                tf.parse_single_sequence_example(serialized=raw
                                                 ,context_features=self.conttext_spec
                                                 ,sequence_features=self.sequence_spec)
            label = context_dic.pop(params["item_relevance"])
            label = label*3
            #seq_dic[self.params["user_car_serial"]] = context_dic[self.params["user_car_serial"]]
            for k,v in six.iteritems(seq_dic):
                context_dic[k] = v
            return context_dic,label

        if mode == tf.estimator.ModeKeys.TRAIN:
            data_set = data_set.repeat(None).map(parse).batch(batch_size)#.prefetch(buffer_size=None)
        elif mode == tf.estimator.ModeKeys.EVAL:
            data_set = data_set.repeat(None) \
                .take(3000).map(parse).batch(60)#.prefetch(buffer_size=None)
        it = data_set.make_one_shot_iterator()
        feature,label = it.get_next()
        label = tf.sparse_tensor_to_dense(label,default_value=-1)
        return feature,label

    def get_input_reciever_fn(self):
        # def serving_input_receiver_fn():
        #     """An input_fn that expects a serialized tf.Example."""
        #     serialized_tf_example = tf.placeholder(
        #         dtype=tf.string,
        #         shape=[None],
        #         name='input_example_tensor')
        #     receiver_tensors = {'examples': serialized_tf_example}
        #     features =serialized_tf_example[0]
        #     self.conttext_spec.pop(self.params["item_relevance"])
        #     context_dic,seq_dic = tf.parse_single_sequence_example(features, self.conttext_spec,self.sequence_spec)
        #     features = {}
        #     for name,feature in context_dic.items():
        #         size = tf.shape(feature)[0]
        #         features[name] = tf.sparse_reshape(feature,[1,size])
        #     for name,feature in seq_dic.items():
        #         shape = tf.shape(feature)
        #         size1 = shape[0]
        #         size2 = shape[1]
        #         features[name] = tf.sparse_reshape(feature,[1,size1,size2])
        #     return ServingInputReceiver(features, receiver_tensors)
        # #serving_input_receiver_fn = build_parsing_serving_input_receiver_fn(self.serving_spec)
        # return serving_input_receiver_fn
        def serving_input_receiver_fn():
            receiver_tensors = dict()
            features = dict()
            for feature_name in self.params["item_feature_list"]:
                input_tensor = tf.placeholder(dtype=tf.string,shape=[None,None],name = feature_name)
                receiver_tensors[feature_name] = input_tensor
                features[feature_name] = input_tensor

            name = self.params["history_item_relevance"]
            receiver_tensors[name] = tf.placeholder(dtype=tf.float32,shape = [None,None],name = name)
            features[name] = receiver_tensors[name]

            name = self.params["user_history_item_title"]
            receiver_tensors[name] = tf.placeholder(dtype=tf.string,shape = [None,None,None],name = name)
            features[name] = receiver_tensors[name]

            name = self.params["item_title"]
            receiver_tensors[name] = tf.placeholder(dtype=tf.string,shape = [None,None,None],name = name)
            features[name] = receiver_tensors[name]

            ret = ServingInputReceiver(features, receiver_tensors)
            return ret
        return serving_input_receiver_fn
