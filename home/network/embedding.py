import six
import tensorflow as tf
import os

from tensorflow import SparseTensor

from home.common.word_embedding import CommonWordEmbedingLayer

'''
  * Created by linsu on 2019/1/25.
  * mailto: lsishere2002@hotmail.com
'''
from tensorflow.python.feature_column.feature_column import categorical_column_with_vocabulary_file, \
    shared_embedding_columns, input_layer, embedding_column, numeric_column
from autohome.common.vocab_file import get_vocab_file_size
class EmbeddingLayer(tf.layers.Layer):
    def __init__(self,params,is_trainning=False, dtype=tf.float32, name="word_embedding"):
        super(EmbeddingLayer, self).__init__(is_trainning, name, dtype)
        self.params = params
        self.columns = {}
        col_dic = {}
        root = params["voc_file_root"]
        for feature_name,voc_file in zip(self.params["item_feature_list"],self.params["item_feature_voc_file"]):
            voc_file = os.path.join(root,voc_file)
            category_col = categorical_column_with_vocabulary_file(key=feature_name,vocabulary_size=get_vocab_file_size(voc_file)
                                                                   ,vocabulary_file=voc_file,num_oov_buckets=1)
            # self.columns[feature_name] = embedding_column(category_col,10)
            if col_dic.get(voc_file):
                col_dic.get(voc_file).append((feature_name,category_col))
            else:
                col_dic[voc_file] = [(feature_name,category_col)]
        for _,value in six.iteritems(col_dic):
            cols=[]
            for name,col in value:
                cols.append(col)
            shared_cols = shared_embedding_columns(cols,dimension=10)
            for (name,_),share_col in zip(value,shared_cols):
                self.columns[name] = share_col

        voc_file =os.path.join(root,self.params["item_title_voc_file"])
        voc_file_size = get_vocab_file_size(voc_file)
        self.common_embedding_layer = CommonWordEmbedingLayer(voc_file_size,voc_file
                                                             ,embedding_size=self.params["embedding_size"]
                                                              ,name = "common_embedding")

        # voc_file = os.path.join(root,self.params["user_car_serial_file"])
        # self.car_serial_col = categorical_column_with_vocabulary_file(key=self.params["user_car_serial"]
        #                                                               ,vocabulary_size=get_vocab_file_size(voc_file)
        #                                                               ,vocabulary_file=voc_file,num_oov_buckets=1)
        # self.car_serial_col = embedding_column(self.car_serial_col,20)

        # self.item_relevance_col = numeric_column(self.params.item_relevance,[1],0)
        # self.item_hot_score_col = numeric_column(self.params["item_hot_score"],[1],0)

    def build(self, _):
        self.built = True

    def call(self, inputs, **kwargs):
        # item_input = {}
        for feature_name in self.params["item_feature_list"]:
            feature = inputs.get(feature_name)
            if isinstance(feature, tf.SparseTensor):
                feature = tf.sparse_tensor_to_dense(feature,default_value='pad')
            shape = tf.shape(feature)
            feature = tf.reshape(feature,[-1,1])
            feature = input_layer({feature_name:feature},[self.columns[feature_name]])
            feature = tf.reshape(feature,[shape[0],shape[1],10])
            inputs[feature_name]=feature
        #[batch_size,history_len,seq_len,embedding_size]
        key = self.params["user_history_item_title"]
        em,len,_,_ = self.common_embedding_layer(inputs[key])
        inputs[key] = em
        inputs["history_item_title_len"] = len
        #[batch_size,recommand_len,seq_len,embedding_size]
        key = self.params["item_title"]
        em,len,_,_ = self.common_embedding_layer(inputs[key])
        inputs[key] = em
        inputs["recommand_title_len"] = len

        # history = inputs[self.params["user_history_item_title"]]
        # recomm  = inputs[self.params["item_title"]]
        #
        # history_len = history.dense_shape[1] if isinstance(history,SparseTensor) else tf.shape(history)[1]
        # recomm_len = recomm.dense_shape[1] if isinstance(recomm,SparseTensor) else tf.shape(recomm)[1]
        #
        # history_recommand = tf.sparse_concat(sp_inputs=[history,recomm] ,axis=1,expand_nonconcat_dim = True) \
        #     if isinstance(history,SparseTensor) else tf.concat([history,recomm] ,axis=1)
        #
        # em,len ,_,_ = self.common_embedding_layer(history_recommand)
        # inputs["history_recommand"] = em
        # inputs["titles_len"] = len
        # inputs["history_len"] = history_len
        # inputs["recomm_len"] = recomm_len

        # inputs[self.params.item_relevance] = input_layer([inputs[self.params.item_relevance]],[self.item_relevance_col])

        # key =self.params["item_hot_score"]
        # value = tf.sparse_tensor_to_dense(inputs[key])
        # inputs[key] = value#input_layer({key:value},[self.item_hot_score_col])
        #
        # key = self.params["user_car_serial"]
        # value = inputs[key]
        # inputs[self.params["user_car_serial"]] = input_layer({key:value},[self.car_serial_col])

        return inputs




