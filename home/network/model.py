import six
import tensorflow as tf

import autohome.rank_lib as tfr
from home.network.embedding import EmbeddingLayer
from home.network.mlp import MLPLayer
from home.network.nlp import NLPLayer
from home.network.resnet import ResLayer

'''
  * Created by linsu on 2019/1/23.
  * mailto: lsishere2002@hotmail.com
'''

tf.flags.DEFINE_string("loss_name","list_mle_loss","The RankingLossKey for loss function.")
tf.flags.DEFINE_string("network_strategy","mlp","res or mlp")

def get_eval_metric_fns():
    """Returns a dict from name to metric functions."""
    metric_fns = {}
    metric_fns.update({
        "metric/%s" % name: tfr.metrics.make_ranking_metric_fn(name) for name in [
        tfr.metrics.RankingMetricKey.ARP,
        tfr.metrics.RankingMetricKey.MRR,
    ]
    })
    metric_fns.update({
        "metric/ndcg@%d" % topn: tfr.metrics.make_ranking_metric_fn(
            tfr.metrics.RankingMetricKey.NDCG, topn=topn)
        for topn in [1, 3, 5, 10]
    })
    return metric_fns

def score_fn(context_features, per_examplefeatures, mode, params, unused_config):
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    if params["network_strategy"] == "res":
        layer = ResLayer(params,is_training)
    elif params["network_strategy"] == "mlp":
        layer = MLPLayer(params,is_training)
    else:
        raise Exception("invalid network_strategy")

    logits = layer([context_features,per_examplefeatures])

    return logits


def transform(feature_dic,mode,params):
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    print("--- model_fn in %s ---" % mode)
    layer = EmbeddingLayer(params,is_training,name = "embedding")
    embedding_features = layer(feature_dic)
    layer = NLPLayer(params,is_training,name = "nlp")
    return layer(embedding_features,mode=mode)


def get_model_fn(params):
    def _train_op_fn(loss):
        """Defines train op used in ranking head."""
        return tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            learning_rate=params["learning_rate"],
            optimizer="Adagrad")

    ranking_head = tfr.head.create_ranking_head(
        loss_fn=tfr.losses.make_loss_fn(params["loss_name"]),
        eval_metric_fns=get_eval_metric_fns(),
        train_op_fn=_train_op_fn)

    return tfr.model.make_groupwise_ranking_fn(
        group_score_fn=score_fn,
        group_size=1,
        transform_fn=transform,
        ranking_head=ranking_head)


