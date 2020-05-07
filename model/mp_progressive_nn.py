import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops
from model.rl_agent import *
from model.nn_primitives import *
from model.progressive_nn import *


class MessagePassingProgressiveNN(ProgressiveNN):

    def __init__(self,
                 emb_size,
                 n_nodes,
                 n_devs,
                 pg,
                 config,
                 mp=False,
                 debug=False,
                 seed=42,
                 normalize_aggs=False,
                 bn_pre_classifier=False,
                 small_nn=False,
                 no_msg_passing=False,
                 radial_mp=None,
                 sage=False,
                 sage_position_aware=False,
                 use_single_layer_perceptron=False,
                 pgnn_c=0.5,
                 pgnn_neigh_cutoff=6,
                 pgnn_anchor_exponent=1,
                 pgnn_aggregation='max',
                 bs=1,
                 dtype=tf.float32):

        ProgressiveNN.__init__(self, seed, config['dont_repeat_ff'], bs)

        self.tri_agg = config['tri_agg']
        E = self.E = tf.placeholder(tf.float32)

        d_msg = emb_size

        if sage:
            position_aware = sage_position_aware
            self.sage = SAGEMessenger(emb_size, d_msg,
                                      sample_ratio=0.5,
                                      hops=2,
                                      position_aware=position_aware,
                                      single_layer_perceptron=use_single_layer_perceptron,
                                      pgnn_c=pgnn_c,
                                      pgnn_neigh_cutoff=pgnn_neigh_cutoff,
                                      pgnn_anchor_exponent=pgnn_anchor_exponent,
                                      pgnn_aggregation=pgnn_aggregation)
            out = self.sage.build(pg, E)
            d_agg = d_msg
        elif not no_msg_passing:
            with tf.variable_scope('Messenger'):
                if radial_mp is not None:
                    self.mess = RadialMessenger(radial_mp,
                                                emb_size,
                                                d_msg,
                                                small_nn=small_nn,
                                                dtype=dtype)
                    out = self.mess.build(pg, pg.get_fadj(), pg.get_badj(), E, bs=bs)
                else:
                    self.mess = Messenger(emb_size, d_msg, small_nn=small_nn)
                    # assert pg.nodes() == pg.topo_order()
                    # out comes out tiled
                    out = self.mess.build(pg, pg.nodes(), E, bs=bs)

            # TODO
            d_agg = 2 * d_msg
            # d_agg = d_msg
            # out = tf.reshape(E, [-1, tf.shape(E)[-1]])
            self.mp_out = out
        else:
            out = tf.reshape(E, [-1, tf.shape(E)[-1]])
            d_agg = d_msg

        if self.tri_agg:
            args = [d_agg, d_agg, d_agg, True, normalize_aggs, small_nn]
            with tf.variable_scope('Parent-Aggregator'):
                self.agg_p = Aggregator(*args)
                agg_p_out = self.agg_p.build(out)
            with tf.variable_scope('Child-Aggregator'):
                self.agg_c = Aggregator(*args)
                agg_c_out = self.agg_c.build(out)
            with tf.variable_scope('Parallel-Aggregator'):
                self.agg_r = Aggregator(*args)
                agg_r_out = self.agg_r.build(out)

            with tf.variable_scope('Self-Embedding'):
                self.self_mask = tf.placeholder(tf.float32, [bs, None])
                self_out = tf.matmul(self.self_mask, out)

            self.agg_p_out, self.agg_c_out, self.agg_r_out = agg_p_out, agg_c_out, agg_r_out
            self.self_out = self_out
            # out = tf.stack([agg_p_out, agg_c_out, agg_r_out, self_out], axis=-1)
            # out = tf.concat([agg_p_out, agg_c_out, agg_r_out, self_out], axis=-1)
            out = [agg_p_out, agg_c_out, agg_r_out, self_out]

            # TODO
            # out = tf.reshape(out, [bs, -1])
            # inp_size = d_agg#out.get_shape()[-1].value
            self.triagg_out = out
            # out = tf.Print(out, [agg_c_out], message='agg_c_out: ', summarize=int(1e6))

        elif config['agg_msgs']:
            # aggregate back and forward message embeddings
            self.agg = Aggregator(d_agg, d_agg, d_agg, False, normalize_aggs,
                                  small_nn)
            print("Here before building message aggregator")
            out = self.agg.build(out)
            print("Here after building message aggregator")

            # TODO
            # out = tf.reshape(out, [bs, -1])
            # inp_size = out.get_shape()[-1].value

        # convert to 1x N shape before feed-forward
        out = tf.reshape(out, [bs, -1])
        inp_size = out.get_shape()[-1].value
        # if inp_size > len(pg.nodes()):
        #     inp_size = len(pg.nodes())

        if bn_pre_classifier:
            out = tf.layers.batch_normalization(out, training=True)

        if small_nn:
            classifier_hidden_layers = [inp_size]
        else:
            # classifier_hidden_layers = [2 * inp_size, inp_size]
            classifier_hidden_layers = [2 * inp_size, inp_size]

        logits = Classifier(inp_size, classifier_hidden_layers, n_devs).build(out)

        if self.tanhc_decay_func is not None:
            logits = tf.tanh(logits) * self.tanhc_decay_func

        self.no_noise_logits = logits
        self.build_train_ops(logits)

    def get_ph(self):
        return self.self_mask

    def get_feed_dict(self, pgs, node, start_times, n_peers):
        if len(pgs) == 1:
            E = pgs[0].get_embeddings()
        else:
            E = []
            for pg in pgs:
                E.append(pg.get_embeddings())

        d = {self.E: E}

        if self.tri_agg:
            bs = len(pgs)
            N = pgs[0].n_nodes()

            def f(mask_fns):
                mask = np.zeros((bs, bs * N), dtype=np.int32)
                for i in range(bs):
                    mask[i, i * N:(i + 1) * N] = mask_fns[i](node)
                return mask

            p_masks = f([pg.get_ancestral_mask for pg in pgs])
            c_masks = f([pg.get_progenial_mask for pg in pgs])
            r_masks = f([
                lambda node: pg.get_peer_mask(node, start_t, n_peers)
                for start_t, pg in zip(start_times, pgs)
            ])
            self_masks = f([pg.get_self_mask for pg in pgs])

            # p_masks = [pg.get_ancestral_mask(node) for pg in pgs]
            # c_masks = [pg.get_progenial_mask(node) for pg in pgs]
            # r_masks = [pg.get_peer_mask(node, start_t, n_peers) for start_t, pg in zip(start_times, pgs)]
            # self_masks = [pg.get_self_mask(node) for pg in pgs]

            for masks in [p_masks, c_masks, r_masks]:
                for m in masks:
                    assert np.all(np.logical_or(m == 0, m == 1))

            d = {
                self.E: np.array(E),
                self.agg_p.get_ph(): np.array(p_masks),
                self.agg_c.get_ph(): np.array(c_masks),
                self.agg_r.get_ph(): np.array(r_masks),
                self.get_ph(): np.array(self_masks)
            }

        return d
