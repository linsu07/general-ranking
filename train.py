import collections

import six
import tensorflow as tf
from tensorflow.contrib.estimator import regression_head
from tensorflow.python.estimator.export import export_lib

'''
  * Created by linsu on 2019/1/21.
  * mailto: lsishere2002@hotmail.com
'''
import sys
from tensorflow.python.estimator.run_config import RunConfig, TaskType
from autohome.common.listeners import EvalListener, LoadEMAHook
from autohome.data import  SparkInput
from autohome.network.model import get_model_fn

flag = tf.flags
tf.flags.DEFINE_string("item_relevance", "item_relevance", "tfrecord中的标签的值")
tf.flags.DEFINE_string("history_item_relevance","history_item_relevance","tfrecord中的历史item的相关性")
tf.flags.DEFINE_string("user_history_item_title","user_history_item_title","title")
tf.flags.DEFINE_string("item_title","item_title","title")
tf.flags.DEFINE_string("item_title_voc_file","item_title_voc_file","file")
tf.flags.DEFINE_list("item_feature_list"
                     ,[
                       "user_history_item_author"
                       # ,"user_history_item_car_serial"
                       ,"user_history_item_type" #文章、图文、话题、视频等
                       # ,"user_history_item_content_type" #新闻、评测、导购等
                       # ,"user_history_item_cmstag"
                       # ,"user_history_item_nlptag"
                        ,"item_author"
                        # ,"item_car_serial"
                        ,"item_type" #文章、图文、话题、视频等
                        # ,"item_content_type" #新闻、评测、导购等
                        # ,"item_cmstag"
                        # ,"item_nlptag"
                       ]
                     ,"推荐品的特征列表")
tf.flags.DEFINE_list("item_feature_voc_file"
                     ,[
                         "item_author_voc_file"
                         # ,"item_car_serial_voc_file"
                         ,"item_type_voc_file" #文章、图文、话题、视频等
                         # ,"item_content_type_voc_file" #新闻、评测、导购等
                         # ,"item_cmstag_voc_file"

                         ,"item_author_voc_file"
                         # ,"item_car_serial_voc_file"
                         ,"item_type_voc_file" #文章、图文、话题、视频等
                         # ,"item_content_type_voc_file" #新闻、评测、导购等
                         # ,"item_cmstag_voc_file"
                     ]
                     ,"推荐品的特征字典文件集合列表")

# tf.flags.DEFINE_string("user_car_serial", "user_car_serial", "用户车系名字")
# tf.flags.DEFINE_string("user_car_serial_file", "car_serial_voc_file", "用户车系名字")

# tf.flags.DEFINE_string("item_hot_score","item_hot_score","推荐项的热度值")

tf.flags.DEFINE_string("voc_file_root","d:\\zeroPrj\\voc","推荐品的特征字典文件根地址")


tf.flags.DEFINE_list("hidden_layer_dims", ["256", "128", "64"],
                  "Sizes for hidden layers.")

tf.flags.DEFINE_integer("lstm_hidden_size",64,"nlp 的lstm cell size")

tf.flags.DEFINE_float("learning_rate", 0.1, '学习率.')
tf.flags.DEFINE_integer("embedding_size",64,"如果使用预先训练的embedding，此参数无效，即embedding_file_path 不为None")
tf.flags.DEFINE_string("embedding_file_path",None,"可选，预训练的embedding文件路径，包括embedding和vocabulary 2个文件，如果不为none，embed_size，feature_voc_file_path参数不起作用")
# tf.flags.DEFINE_string("feature_voc_file_path", None, "tfrecord中的特征词的字典文件地址，为了兼容spark，目录下唯一text为file")
tf.flags.DEFINE_list("gpu_cores",None,"例如[0,1,2,3]，在当个GPU机器的情况，使用的哪些核来训练")
tf.flags.DEFINE_string('log_level', 'INFO', 'tensorflow训练时的日志打印级别， 取值分别为，DEBUG，INFO,WARN,ERROR')
tf.flags.DEFINE_string('data_dir', 'D:\\zeroPrj\\tf_UserHistory', '训练数据存放路径，支持hdfs')
tf.flags.DEFINE_string('model_dir', 'd:\\zeroPrj\\model\\', '保存dnn模型文件的路径，支持hdfs')
tf.flags.DEFINE_integer('batch_size', 3, '一批数量样本的数量')
tf.flags.DEFINE_integer("check_steps", 10,'保存训练中间结果的间隔，也是evalutation的间隔')
tf.flags.DEFINE_integer('max_steps', 10, '训练模型最大的批训练次数，在model_dir不变的情况下重复训练'
                                               '，达到max_step后，不再继续训练，或者增加max_step，或者更换model_dir, 再继续训练')
tf.flags.DEFINE_float("drop_out_rate", 0.1, "dropout概率，范围是0至1。例如rate=0.1会将输入Tensor的内容dropout10%。")
tf.flags.DEFINE_integer("enable_ema",0,"是否启动指数移动平均来计算参数")
tf.flags.DEFINE_float("ema_decay",0.99,"ema的decay速率")
tf.flags.DEFINE_string("elmo_ckpt_path", "", "elmo预训练模型check_point地址")

params ={}

def main(_):
    # 配置日志等级
    level_str = 'tf.logging.{}'.format(str(tf.flags.FLAGS.log_level).upper())
    tf.logging.set_verbosity(eval(level_str))
    params = flag.FLAGS.flag_values_dict()

    # 加载数据
    input = SparkInput(params)
    # estimator运行环境配置
    session_config = tf.ConfigProto()
    session_config.allow_soft_placement = True
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    session_config.gpu_options.allow_growth = True
    # session_config.log_device_placement = True

    if flag.FLAGS.gpu_cores:
        gpu_cors = tuple(flag.FLAGS.gpu_cores)
        devices =  ["/device:GPU:%d" % int(d) for d in gpu_cors]
        tf.logging.warn("using device: " + " ".join(devices))
        distribution = tf.contrib.distribute.MirroredStrategy(devices = devices)

        tf.logging.warn("in train.py, distribution")
        tf.logging.warn(distribution._devices)
        config = RunConfig(save_checkpoints_steps=params.check_steps,train_distribute=distribution, keep_checkpoint_max=2, session_config=session_config)
    else:
        config = RunConfig(save_checkpoints_steps=flag.FLAGS.check_steps, keep_checkpoint_max=2, session_config=session_config)

    # def model_fn(features,labels,mode:tf.estimator.ModeKeys,config: RunConfig, params):
    #     feature = features["item_author"]
    #     if isinstance(feature,tf.SparseTensor):
    #         feature = tf.sparse_tensor_to_dense(feature,"example")
    #     logit = tf.zeros_like(feature)
    #     head = regression_head()
    #     def _train_op_fn(loss):
    #         """Defines train op used in ranking head."""
    #         return tf.contrib.layers.optimize_loss(
    #             loss=loss,
    #             global_step=tf.train.get_global_step(),
    #             learning_rate=params["learning_rate"],
    #             optimizer="Adagrad")
    #     spec = head.create_estimator_spec(features,mode,logits=logit,labels=labels
    #                                ,train_op_fn=_train_op_fn)
    #
    #     spec.export_outputs['serving_default'] = export_lib.PredictOutput(logit)
    #     return spec


    estimator = tf.estimator.Estimator(model_fn=get_model_fn(params), model_dir=flag.FLAGS.model_dir, config=config, params=params)
    #estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=flag.FLAGS.model_dir, config=config, params=params)

    train_data_dir = input.get_data_dir(tf.estimator.ModeKeys.TRAIN, params)
    eval_data_dir = input.get_data_dir(tf.estimator.ModeKeys.EVAL, params)

    hook = [] if not flag.FLAGS.enable_ema else [LoadEMAHook(flag.FLAGS.model_dir,0.99)]

    listeners = [
        EvalListener(estimator, lambda: input.input_fn(mode = tf.estimator.ModeKeys.EVAL, params=params, data_dir=train_data_dir), name="train_data",hook = hook),
        EvalListener(estimator, lambda: input.input_fn(mode = tf.estimator.ModeKeys.EVAL, params=params, data_dir=eval_data_dir),hook = hook)
    ]

    # pre-trained elmo embedding need a hook to load variables
    train_hooks = []
    # if params.elmo_ckpt_path and tf.gfile.Exists(params.elmo_ckpt_path):
    #     print("!!! load elmo check point path: " + params.elmo_ckpt_path)
    #     train_hooks += [ELMoWarmStartHook(params.elmo_ckpt_path)]

    def train_input_fn():
        return input.input_fn(mode = tf.estimator.ModeKeys.TRAIN, params=params, data_dir=train_data_dir)
    from tensorflow.python import debug as tf_debug
    debug_hook = [tf_debug.LocalCLIDebugHook(ui_type="readline")]
    #estimator.train(train_input_fn, max_steps=flag.FLAGS.max_steps, saving_listeners=listeners, hooks=debug_hook)
    estimator.train(train_input_fn, max_steps=flag.FLAGS.max_steps, saving_listeners=listeners, hooks=train_hooks)
    dir = estimator.export_savedmodel(tf.flags.FLAGS.model_dir, input.get_input_reciever_fn())
    tf.logging.warn("save model to %s" % dir)



if __name__ == "__main__":
    tf.app.run(main, argv=None)


