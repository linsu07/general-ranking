# encoding:utf-8
import six

import train
import tensorflow as tf

from autohome.data import SparkInput
from autohome.network.model import transform, get_model_fn


import math

def basic_sigmoid(x):
    s = 1 / (1+ math.exp(-x))
    return s


# 点击、-0.5
# 赞、   0
# 评论、0.5
# 转发、2.0
# 收藏、3.0

input = [-0.5,0,0.5,2.0,3.0]

output = [basic_sigmoid(x) for x in input]
print(output)


# tf.enable_eager_execution()
# params = tf.flags.FLAGS.flag_values_dict()

# input = SparkInput(params)
# inputs = \
#     '{"user_history_item_car_serial":["奥迪A4","宝马5系","no"],"user_history_item_author":["Linux","李","wang"],"user_history_item_cmstag":["车展","对比","旅游"],"user_history_item_content_type":["新闻","评测","导购"],"user_history_item_title":["我们怎么去了a这个cc地方","你一定敢兴趣","这个文章好看"],"user_history_item_type":["文章","图文","视频"],"item_author":["张","li","文"],"item_type":["图文","视频","视频"],"item_content_type":["新闻","评测","导购"],"item_title":["深深的海洋","北京大兴变的人多起来了","海洋的北京"],"item_cmstag":["车展","对比","旅游"],"item_car_serial":["奥迪A4","宝马5系","no"]}'
# inputs = tf.constant(inputs)
# spec = input.serving_spec
# ret = tf.parse_single_example(inputs,spec)
#
# print (ret)




# train_data_dir = input.get_data_dir(tf.estimator.ModeKeys.TRAIN, params)
# feature,label = input.input_fn(mode = tf.estimator.ModeKeys.TRAIN, params=params, data_dir=train_data_dir)
# # it = data_set.make_one_shot_iterator()
# # feature,label = it.get_next()
# # context,per_example = transform(feature,tf.estimator.ModeKeys.TRAIN,params)
#
# model_fn= get_model_fn(params)
#
# model_fn(feature,label,mode = tf.estimator.ModeKeys.TRAIN,params= params,config=None)

# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     iterator = data_set.make_initializable_iterator()
#
#     sess.run(iterator.initializer)
#     it = data_set.make_one_shot_iterator()
#     feature,label = it.get_next()
#
#     # context,per_example = transform(feature,tf.estimator.ModeKeys.TRAIN,params)
#     for k,v in six.iteritems(feature):
#             print(k)
#             value, = sess.run([v])
#             print(value)
#     l,= sess.run([label])
#     print(l)


