
import tensorflow as tf
from tensorflow import sparse_tensor_to_dense
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn, dynamic_rnn

'''
  * Created by linsu on 2019/1/24.
  * mailto: lsishere2002@hotmail.com
'''
tf.flags.DEFINE_integer("enable_his_relevance_weight",0,"是否启动his_relevance_weight ")
tf.flags.DEFINE_string("history_strategy","sum","对待用户历史的叠加策略")

class NLPLayer(tf.layers.Layer):
    def __init__(self,params,is_trainning=False, dtype=tf.float32, name="nlp"):
        super(NLPLayer, self).__init__(is_trainning, name, dtype)
        self.params = params

    def build(self, _):
        self.fw_cell = tf.nn.rnn_cell.LSTMCell(
            self.params["lstm_hidden_size"]
            ,initializer=xavier_initializer()
            ,name = "fw_cell"
        )
        # self.bw_cell = tf.nn.rnn_cell.LSTMCell(
        #     self.params["lstm_hidden_size"]
        #     ,initializer=xavier_initializer()
        #     ,name = "bw_cell"
        # )
        self.built = True

    def get_lstm_embedding(self, feature,feature_len,name):
        #[batch_size,history_len,seq_len,embedding_size]
        shape = tf.shape(feature)
        batch_size = shape[0]
        history_len = shape[1]
        seq_len = shape[2]
        #[batch_size,history_len]
        # history_item_len = input["history_item_title_len"]
        # #[batch_size,1]
        # user_history_size = tf.reduce_sum(tf.sign(history_item_len),-1,keep_dims=True)
        feature_len = tf.reshape(feature_len,[-1])

        items = tf.reshape(feature,[batch_size*history_len,seq_len,self.params["embedding_size"]])

        # _,(output_state_fw, output_state_bw) = bidirectional_dynamic_rnn(self.fw_cell
        #                                                   ,self.bw_cell
        #                                                   ,items
        #                                                   ,sequence_length=feature_len
        #                                                   ,dtype=tf.float32
        #                                                   ,scope="dy_rnn_{}".format(name))
        # state = [output_state_fw.c,output_state_bw.c]
        # title_embedding = tf.concat(state,-1)
        # title_embedding = tf.reshape(title_embedding,[batch_size,history_len,self.params["lstm_hidden_size"]*2])
        title_embedding = tf.reduce_sum(items,axis=-2)
       # _, state = dynamic_rnn(self.fw_cell,items,feature_len,dtype=tf.float32,scope="dy_rnn")
        #title_embedding = state.c
        title_embedding = tf.reshape(title_embedding,[batch_size,history_len,self.params["lstm_hidden_size"]])

        return title_embedding,batch_size,history_len

    def call(self, inputs, **kwargs):
        mode = kwargs.get('mode')
        # context_features = {}
        #[batch_size,his_len,2*hidden_size]
        history = inputs[self.params["user_history_item_title"]]
        recomm = inputs[self.params["item_title"]]

        # history_recommand = inputs["history_recommand"]
        # history_recommand_title_len = inputs["titles_len"]
        # history_len = inputs["history_len"]
        # recommd_size = inputs["recomm_len"]
        #
        # title_embedding,_,_ =  self.get_lstm_embedding(history_recommand
        #                                            ,history_recommand_title_len,name = "title_embedding")
        # embedding_list = tf.split(title_embedding,axis = 1,num_or_size_splits=[history_len,recommd_size])
        #
        # len_list = tf.split(history_recommand_title_len,axis = 1,num_or_size_splits=[history_len,recommd_size])
        # history_item_title_len = len_list[0]
        # recommand_title_len = len_list[1]


        # title_his_embedding = embedding_list[0]

        history_item_title_len = inputs["history_item_title_len"]
        recommand_title_len = inputs["recommand_title_len"]

        if mode == tf.estimator.ModeKeys.PREDICT:
            his_size = tf.shape(history)[1]
            rec_size = tf.shape(recomm)[1]
            all = tf.concat([history,recomm],axis=1)
            len = tf.concat([history_item_title_len,recommand_title_len],axis=1)
            title_embedding,_,_ = self.get_lstm_embedding(all,len,name="all_embedding")
            embedding_list = tf.split(title_embedding,num_or_size_splits = [his_size,rec_size],axis=1)
            title_his_embedding = embedding_list[0]
            title_recom_embedding =embedding_list[1]
            recommd_size = rec_size
        else:
            title_his_embedding,_,_ = self.get_lstm_embedding(history
                                                              ,history_item_title_len,name = "user_history")
            title_recom_embedding,_,recommd_size = self.get_lstm_embedding(recomm
                                                                           ,recommand_title_len
                                                                           ,name = "recomm_item")


        user_mask = tf.cast(tf.expand_dims(tf.sign(history_item_title_len),-1),tf.float32)
        user_history_size = tf.reduce_sum(user_mask,1)

        user_feature = [title_his_embedding]
        for feature_name in self.params["item_feature_list"]:
            if str(feature_name).startswith("user_history"):
                user_feature.append(inputs[feature_name])

        user_feature = tf.concat(user_feature,axis=-1)
        user_feature = user_feature*user_mask


        # title_recom_embedding= embedding_list[1]
        #[batch_size,recomm_size,2*hidden_size]



        item_mask = tf.cast(tf.expand_dims(tf.sign(recommand_title_len),-1),tf.float32)
        item_feature = [title_recom_embedding]
        for feature_name in self.params["item_feature_list"]:
            if str(feature_name).startswith("item_"):
                item_feature.append(inputs[feature_name])

        item_feature = tf.concat(item_feature,-1)*item_mask

        his_item_weight = inputs["history_item_relevance"]
        if isinstance(his_item_weight, tf.SparseTensor):
            his_item_weight = sparse_tensor_to_dense(his_item_weight,default_value=0.0)
        #[batch_size,his_size,1]
        his_item_weight = tf.expand_dims(his_item_weight,axis=-1)

        if self.params["history_strategy"]=="sum":
            if self.params["enable_his_relevance_weight"]:
                user_feature = user_feature*his_item_weight
                user_weighted_size = tf.reduce_sum(his_item_weight,axis=1)
                user_feature = tf.reduce_sum(user_feature,axis = 1)/(user_weighted_size+1e-8)
            else:
                user_feature = tf.reduce_sum(user_feature,axis = 1)/(user_history_size+1e-8)

            #[batch_size,1,item_size]
            user_feature = tf.expand_dims(user_feature,1)
            user_feature = tf.tile(user_feature,multiples=[1,recommd_size,1])
            user_item_feature = tf.concat([user_feature,item_feature],-1)

        elif self.params["history_strategy"] == "attention":
            if self.params["enable_his_relevance_weight"]:
                user_feature = user_feature*his_item_weight
            #[batch_size,recomm_size,his_len]
            factor = tf.matmul(item_feature,user_feature,transpose_b=True)
            #[batch_size,recomm_size,item_size]
            total_user_feature = tf.matmul(factor,user_feature)

            user_item_feature = tf.concat([total_user_feature,item_feature],-1)
        else:
            raise Exception("wrong history_strategy config")

        context_features = {
            # self.params["user_car_serial"]:inputs[self.params["user_car_serial"]]
        }
        per_example_features = {
            "user_item_list_features":user_item_feature,
            # self.params["item_hot_score"]:inputs[self.params["item_hot_score"]]
        }
        return context_features,per_example_features







