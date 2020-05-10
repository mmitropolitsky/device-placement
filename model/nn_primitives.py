import random
import traceback

import networkx as nx
import numpy as np
import tensorflow as tf
import datetime
from tensorflow.python.framework import ops as tf_ops

INIT_SCALE = 1


def glorot(shape, scope='default', dtype=tf.float32):
    # Xavier Glorot & Yoshua Bengio (AISTATS 2010) initialization (Eqn 16)
    with tf.variable_scope(scope):
        init_range = np.sqrt(6.0 * INIT_SCALE / (shape[0] + shape[1]))
        init = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=dtype)
        return tf.Variable(init, dtype=dtype)


def zero_init(shape, scope='default', dtype=tf.float32):
    with tf.variable_scope(scope):
        init = np.zeros(shape)
        return tf.Variable(init, dtype=dtype)


class SingleLayerFNN(object):
    def __init__(self, inp_size, inp_shape, name):
        self.w = glorot(inp_shape, name)
        self.b = zero_init((1, inp_size), name)

    def build(self, input_tensor):
        out = input_tensor
        out = tf.matmul(out, self.w) + self.b
        out = tf.nn.relu(out)
        return out


class FNN(object):
    # hidden_layer_sizes: list of hidden layer sizes
    # out_size: Size of the last softmax layer
    def __init__(self, inp_size, hidden_layer_sizes, out_size, name, dtype=tf.float32):

        layers = []
        sizes = [inp_size] + hidden_layer_sizes + [out_size]
        for i in range(len(sizes) - 1):
            w = glorot((sizes[i], sizes[i + 1]), name, dtype=dtype)
            b = zero_init((1, sizes[i + 1]), name, dtype=dtype)
            layers.append([w, b])

        self.layers = layers

    # *Don't* add softmax or relu at the end
    def build(self, inp_tensor):
        out = inp_tensor
        for idx, [w, b] in enumerate(self.layers):
            out = tf.matmul(out, w) + b
            if idx != len(self.layers) - 1:
                out = tf.nn.relu(out)

        return out


'''
  Combines a bunch of embeddings together at a specific node
  g(sum_i(f_i)) = relu_g(Sum_i(relu_f(e_i* M_f + b_f))* M_g + b_g)
  g(sum_i(f_i)) = relu_g((Mask* (relu_f(E* M_f + b_f)))* M_g + b_g)
  To be more specific when the number of embeddings to combine is variable,
  we use a mask Ma
  Dimensions:
    E: N x d        placeholder
    M_f: d x d1     Variable
    b_f: 1 x d1     Variable
    Ma: 1 x N       placeholder
    M_g: d1 x d2    Variable
    b_g: 1 x d2     Variable
'''


class Aggregator(object):
    # N is the max number of children to be aggregated
    # d is the degree of embeddings
    # d1 is the degree of embedding transformation
    # d2 is degree of aggregation
    def __init__(self, d, d1=None, d2=None, use_mask=True, normalize_aggs=False,
                 small_nn=False, dtype=tf.float32):
        self.d = d
        self.d1 = d1
        self.d2 = d2
        self.normalize_aggs = normalize_aggs
        self.dtype = dtype

        if d1 is None:
            d1 = self.d1 = d
        if d2 is None:
            d2 = self.d2 = d

        self.use_mask = use_mask
        if use_mask:
            self.Ma = tf.placeholder(dtype, shape=(None, None))

        if small_nn:
            hidden_layer_f, hidden_layer_g = [], []
        else:
            hidden_layer_f = [self.d]
            hidden_layer_g = [self.d1]

        self.f = FNN(self.d, hidden_layer_f, self.d1, 'f', dtype=dtype)
        self.g = FNN(self.d1, hidden_layer_g, self.d2, 'g', dtype=dtype)

    def build(self, E, debug=False, mask=None):
        summ = 100

        f = tf.nn.relu(self.f.build(E))

        self.f_out = f

        if debug:
            f = tf.Print(f, [f], message='output of f: ', summarize=summ)

        if self.use_mask or mask is not None:
            if mask is None:
                mask = self.Ma

            g = tf.matmul(mask, f)
            if self.normalize_aggs:
                d = tf.cond(
                    tf.reduce_sum(mask) > 0,
                    lambda: tf.reduce_sum(mask),
                    lambda: 1.)

                g /= d

            if debug:
                print(f, g, self.Ma)
        else:
            g = tf.reduce_sum(f, 0, keepdims=True)

        if debug:
            g = tf.Print(g, [g], message='after mask: ', summarize=summ)

        g = tf.nn.relu(self.g.build(g))

        if debug:
            g = tf.Print(g, [g], message='output of g: ', summarize=summ)

        return g

    def get_ph(self):
        return self.Ma


class Classifier(object):
    def __init__(self, inp_size, hidden_layer_sizes, out_size, dtype=tf.float32):
        self.nn = FNN(inp_size, hidden_layer_sizes,
                      out_size, 'classifier', dtype=dtype)

    def build(self, inp_tensor):
        return self.nn.build(inp_tensor)


class SAGEMessenger(object):
    """
    Implementation of GraphSAGE-like algorithm for embedding to be used in the RL policy.

    Paper: "Inductive Representation Learning on Large Graphs" (https://arxiv.org/pdf/1706.02216.pdf)

    Parameters:

    - `embedding_size` - int - degree of embeddings
    - `embedding_transformation_deg` - int
    - `small_nn` - currently not used
    - `sample_ratio` - float [0,1] - what part of a node's neighbours are used to calculate its embeddings
    - `hops` - int [1,2] - how many hops away need to be aggregated
    - `aggregation` - {'mean', 'max', 'min', 'sum'} - how are a node's neighbours aggregated. Default is 'mean'
    """
    def __init__(self, embedding_size_deg,
                 embedding_transformation_deg,
                 small_nn=False,
                 sample_ratio=0.5,
                 hops=2,
                 aggregation='mean',
                 position_aware=True,
                 single_layer_perceptron=False,
                 pgnn_c=0.5,
                 pgnn_neigh_cutoff=6,
                 pgnn_anchor_exponent=1,
                 pgnn_aggregation='max',
                 dtype=tf.float32):
        self.sample_ratio = sample_ratio
        self.embedding_size_deg = embedding_size_deg
        self.embedding_transformation_deg = embedding_transformation_deg
        self.small_nn = small_nn
        self.hops = hops
        self.aggregation = aggregation
        self.position_aware = position_aware
        self.single_layer_perceptron = single_layer_perceptron

        self.pgnn_c = pgnn_c
        self.pgnn_neigh_cutoff = pgnn_neigh_cutoff
        self.pgnn_anchor_exponent = pgnn_anchor_exponent
        self.pgnn_aggregation = pgnn_aggregation

        self.memo = {}

        self.samples = {}
        self.anchor_sets = []
        self.fnns = {}
        self.distances = {}

        self._init_fnns()

    def _init_fnns(self):
        if self.single_layer_perceptron:
            with tf.name_scope('self_transform'):
                self.self_transform = SingleLayerFNN(inp_size=self.embedding_size_deg,
                                                     inp_shape=(self.embedding_size_deg, self.embedding_size_deg),
                                                     name='self_transform')
            for i in range(self.hops + 1):
                current_scope = 'shared' + str(i)
                with tf.name_scope(current_scope):
                    self.fnns[i] = SingleLayerFNN(inp_size=self.embedding_size_deg,
                                                  inp_shape=(self.embedding_size_deg, self.embedding_size_deg),
                                                  name=current_scope)
            with tf.name_scope('positional_awareness'):
                self.fnns['pos'] = SingleLayerFNN(inp_size=self.embedding_size_deg,
                                                  inp_shape=(self.embedding_size_deg, self.embedding_size_deg),
                                                  name='positional_awareness')
        else:
            with tf.name_scope('self_transform'):
                self.self_transform = FNN(hidden_layer_sizes=[self.embedding_size_deg],
                                          inp_size=self.embedding_size_deg,
                                          out_size=self.embedding_transformation_deg,
                                          name='self_transform')
            for i in range(self.hops + 1):
                current_scope = 'shared' + str(i)
                with tf.name_scope(current_scope):
                    self.fnns[i] = FNN(inp_size=self.embedding_size_deg,
                                       hidden_layer_sizes=[self.embedding_size_deg * (i + 1)],
                                       out_size=self.embedding_transformation_deg,
                                       name=current_scope)

            with tf.name_scope('positional_awareness'):
                self.fnns['pos'] = FNN(hidden_layer_sizes=[self.embedding_size_deg],
                                       inp_size=self.embedding_size_deg,
                                       out_size=self.embedding_transformation_deg,
                                       name='positional_awareness')

    def build(self, G, E, bs=1):
        """
        Build embeddings for the nodes of a graph similar to GraphSAGE
        """

        """
        1. Build an FNN with the E placeholder
        """
        self.self_transform = self.self_transform.build(E)

        """
        2. Generate samples of each node's neighbourhood. Based on the `sample_ratio` class parameter
        """
        self._generate_samples(G)

        """
        3. Given that we have the of each node and its neighbourhood sample, we generate embeddings for nodes located 
        n hops away.
        """
        for i in range(0, self.hops + 1):
            for n in G.nodes():
                """
                3.1. Get the embeddings of the current node and its neighbours
                """
                embedding, neighbour_embeddings = self._get_embeddings(G, n, i)

                if len(neighbour_embeddings) > 0:
                    """
                    3.2. We aggregate the neighbours of the current node. The aggregation options are:
                    - Mean (default) (tf.reduce_mean)
                    - Max (tf.reduce_max)
                    - Min (tf.reduce_min)
                    - Sum (tf.reduce_sum)
                    """
                    embedding = tf.transpose(embedding)
                    neighbor_aggregation = [tf.transpose(neigh) for neigh in neighbour_embeddings]
                    neighbor_aggregation = self._aggregate_for_node(neighbor_aggregation)

                    """
                    3.3. Concatenate the neighbourhood aggregation with the embedding of the current node
                    """
                    concatenated_with_current = tf.reshape(tf.concat((neighbor_aggregation, embedding), axis=1),
                                                           shape=[-1, self.embedding_size_deg])
                    """
                    3.5. Generate embedding for the concatenation
                    """
                    embedding = self.fnns[i].build(concatenated_with_current)
                else:
                    embedding = tf.pad(self.fnns[i].build(embedding), paddings=[[2 ** i, 0], [0, 0]])

                """
                3.6. Add the generated embedding to the dictionary so it can be used in the next hop.
                """
                # TODO this might save some time?
                # if i != 0:
                #     del self.samples[n][str(i - 1)]

                if i != self.hops:
                    embedding = tf.nn.dropout(embedding, rate=0.5)
                    self.samples[n][str(i + 1)] = embedding
                else:
                    embedding = tf.nn.l2_normalize(embedding)

                    self.samples[n][str(i)] = embedding
                    self.samples[n][str(i) + 'pooled'] = tf.squeeze(tf.nn.pool(tf.expand_dims(embedding, axis=0),
                                                                               window_shape=[
                                                                                   embedding.shape[-1].value / 2 + 1],
                                                                               pooling_type='AVG', padding='VALID'))
        """
        4. Return the concatenated node embeddings for all nodes for the given number of hops
        """
        if self.position_aware:

            self._precalculate_distances(G.G, self.pgnn_neigh_cutoff)

            self._build_anchor_sets(G)

            positional_info_generator = self._aggregate_positional_info(G.nodes(), self.pgnn_aggregation)

            positions = [pos for pos in positional_info_generator]
            # positions = tf.concat(positions, axis=0)

            out = tf.reshape(positions, shape=[-1, self.embedding_transformation_deg])
            print(out.shape)
            print("Returning P-GNN values", datetime.datetime.now())
            out = tf.identity(out, name='pgnn')
            return out

        else:
            print('Returning GraphSAGE concatenated values.', datetime.datetime.now())
            return tf.concat([self.samples[n][str(self.hops)] for n in G.nodes()], axis=0)

    def _aggregate_positional_info(self, nodes, aggregation='max'):
        print("P-GNN aggregation is", aggregation)
        for i, n in enumerate(nodes):
            if self.memo.get(n) is None:
                self.memo[n] = {}
            print(i, n)
            positional_aggregation = []
            for anchor_set in self.anchor_sets:
                aggregated = None
                if aggregation == 'max':
                    aggregated = self._max_aggregate_anchor(anchor_set, n)
                elif aggregation == 'mean':
                    # This one has a big performance overhead
                    aggregated = self._mean_aggregate_anchor(anchor_set, n)
                positional_aggregation.append(aggregated)

            positional_aggregation = tf.concat(positional_aggregation, axis=0)
            positional_aggregation = tf.reduce_mean(positional_aggregation, axis=0)
            positional_aggregation = tf.expand_dims(positional_aggregation, axis=0)
            yield self.fnns['pos'].build(positional_aggregation)

    def _mean_aggregate_anchor(self, anchor_set, node):
        node_positions = []
        for anchor in anchor_set:
            if self.memo[node].get(anchor) is not None:
                node_anchor_relation = self.memo[node][anchor]
                node_positions.append(node_anchor_relation)
                continue

            node_embedding = self.samples[node][str(self.hops) + 'pooled']
            anchor_embedding = self.samples[anchor][str(self.hops) + 'pooled']

            # positional info between n and anchor node
            if self.distances.get(node) is not None and self.distances[node].get(anchor) is not None:
                positional_info = 1 / (self.distances[node][anchor] + 1)
                feature_info = tf.concat((node_embedding, anchor_embedding), axis=0)
                node_anchor_relation = positional_info * feature_info
            else:
                node_anchor_relation = tf.zeros(shape=[node_embedding.shape[0] + anchor_embedding.shape[0],
                                                       node_embedding.shape[-1]])

            self.memo[node][anchor] = node_anchor_relation
            node_positions.append(node_anchor_relation)
        return tf.reduce_mean(node_positions, axis=0)

    def _max_aggregate_anchor(self, anchor_set, node):

        anchor_node_intersections = [(k, self.distances[node][k]) for k in anchor_set
                                     if self.distances[node].get(k) is not None and k != node]
        max_agg_anchor = max(anchor_node_intersections, key=lambda i: i[1], default=None)
        node_embedding = self.samples[node][str(self.hops) + 'pooled']
        if max_agg_anchor is None:
            return tf.zeros(shape=[node_embedding.shape[0] + node_embedding.shape[0],
                                   node_embedding.shape[-1]])
        anchor_embedding = self.samples[max_agg_anchor[0]][str(self.hops) + 'pooled']
        positional_info = 1 / (self.distances[node][max_agg_anchor[0]] + 1)
        feature_info = tf.concat((node_embedding, anchor_embedding), axis=0)
        node_anchor_relation = positional_info * feature_info
        return node_anchor_relation

    def _generate_samples(self, G):
        for n in G.nodes():
            if self.samples.get(n) is None:
                self.samples[n] = {}
            self.samples[n]['init'] = self._get_sample(G, n)

    def _get_embeddings(self, G, n, level):
        neigh_samples = self.samples[n]['init']
        # if we don't have the initial embeddings of current node and its neighbours

        # TODO maybe move to a separate method?
        """
        3.1.1 If it is the first level, we generate the embedding based only on the fetures of each node.
        We return the embedding of the current node for the current level, as well as the list of its neighbours'
        embeddings for the same level.
        """
        if level == 0:
            self._generate_initial_self_embedding(G, n, self.self_transform)
            for ns in neigh_samples:
                self._generate_initial_self_embedding(G, ns, self.self_transform)

        return self.samples[n][str(level)], [self.samples[neigh][str(level)] for neigh in neigh_samples]

    def _generate_initial_self_embedding(self, G, node, n_transform):
        if self.samples[node].get('0') is None:
            self.samples[node]['0'] = tf.expand_dims(n_transform[G.get_idx(node), :], axis=0)

    def _aggregate_for_node(self, aggregated):
        aggregation = self.aggregation
        if aggregation == 'mean':
            aggregated = tf.reduce_mean(aggregated, axis=0)
        elif aggregation == 'max':
            aggregated = tf.reduce_max(aggregated, axis=0)
        elif aggregation == 'min':
            aggregated = tf.reduce_min(aggregated, axis=0)
        elif aggregation == 'sum':
            aggregated = tf.reduce_sum(aggregated, axis=0)

        return aggregated

    def _get_sample(self, G, node):
        """
        Get a random sample of neighbours based on the ratio
        e.g. if the ratio is 0.5, we will return only half the successors of the current node
        """
        neighbors = [neighbor for neighbor in G.neighbors(node)]
        samples = random.sample(neighbors, int(len(neighbors) * self.sample_ratio))
        return samples

    def _precalculate_distances(self, G, cutoff=6):
        self.distances = dict(nx.all_pairs_shortest_path_length(G, cutoff))

    def _build_anchor_sets(self, G, c=0.2):
        n = len(G.nodes())
        m = int(np.log(n))
        copy = int(self.pgnn_c * m)
        for i in range(m):
            anchor_size = int(n / np.exp2(i + self.pgnn_anchor_exponent))
            for j in range(np.maximum(copy, 1)):
                self.anchor_sets.append(random.sample(G.nodes(), anchor_size))
        print("Number of anchor sets: ", len(self.anchor_sets),
              ". Biggest set is:" + str(int(n / np.exp2(self.pgnn_anchor_exponent))))

    ''' 
      change this to be generic across different graphs to be placed
    '''


class Messenger(object):

    def __init__(self, d, d1, small_nn=False, dtype=tf.float32):
        # forward pass
        with tf.name_scope('FPA'):
            # self.fpa = Aggregator(d, d1, d1, False, small_nn=small_nn, dtype=dtype)
            self.fpa = Aggregator(d, d1, d1, False, small_nn=small_nn, dtype=dtype)
        with tf.name_scope('BPA'):
            self.bpa = Aggregator(d, d1, d1, False, small_nn=small_nn, dtype=dtype)
            # self.bpa = Aggregator(d, d1, d1, False, small_nn=small_nn, dtype=dtype)
        with tf.name_scope('node_transform'):
            if small_nn:
                self.node_transform = FNN(d, [d], d1, 'fnn', dtype=dtype)
            else:
                self.node_transform = FNN(d, [d, d], d1, 'fnn', dtype=dtype)

    def build(self, G, node_order, E, bs=1):
        try:
            self_trans = self.node_transform.build(E)

            def message_pass(nodes, messages_from, agg):
                node2emb = {}

                for n in nodes:
                    msgs = [node2emb[pred] for pred in messages_from(n)]

                    node2emb[n] = tf.expand_dims(self_trans[G.get_idx(n), :],
                                                 axis=0)

                    if len(msgs) > 0:
                        t = tf.concat(msgs, axis=0)
                        inp = agg.build(t)
                        node2emb[n] += inp

                return tf.concat([node2emb[n] for n in G.nodes()], axis=0)
                # TODO
                # return [node2emb[n] for n in G.nodes()]

            out_fpa = message_pass(node_order, G.predecessors, self.fpa)
            print("Finished forward pass of Messenger")
            out_bpa = message_pass(reversed(node_order), G.neighbors, self.bpa)
            print("Finished backward pass of Messenger")

            out = tf.concat([out_fpa, out_bpa], axis=-1)
            # TODO
            # out = [out_fpa, out_bpa]
            # out = tf.reduce_mean(tf.concat([out_bpa,out_fpa], axis=0), axis=0)
            return out
        except Exception:
            # import my_utils; my_utils.PrintException()
            traceback.print_exc()
            # import pdb
            # pdb.set_trace()


class RadialMessenger(Messenger):

    def __init__(self, k, d, d1, small_nn=False, dtype=tf.float32):
        Messenger.__init__(self, d, d1, small_nn, dtype)
        self.dtype = dtype
        self.k = k

    def build(self, G, f_adj, b_adj, E, bs=1):
        assert np.trace(f_adj) == 0
        assert np.trace(b_adj) == 0

        E = tf.cast(E, dtype=self.dtype)

        E = tf.reshape(E, [-1, tf.shape(E)[-1]])
        self_trans = self.node_transform.build(E)

        # self_trans = tf.Print(self_trans, [self_trans], message='self_trans: ', summarize=100000000)

        def message_pass(adj, agg):
            sink_mask = (np.sum(adj, axis=-1) > 0)
            # sink_mask = np.float32(sink_mask)
            # sink_mask = np.float16(sink_mask)
            # adj = np.float16(adj)
            sink_mask = tf.cast(sink_mask, self.dtype)
            adj = tf.cast(adj, self.dtype)

            x = self_trans
            for i in range(self.k):
                # x = tf.Print(x, [x], message='pre agg: x', summarize=1000)
                x = agg.build(x, mask=adj)
                # x = tf.Print(x, [x], message='x', summarize=1000)
                x = sink_mask * tf.transpose(x)
                x = tf.transpose(x)
                x += self_trans

            return x

        def f(adj):
            n = adj.shape[0]
            t = np.zeros([bs * n] * 2, dtype=np.float32)
            for i in range(bs):
                t[i * n: (i + 1) * n, i * n: (i + 1) * n] = adj

            return t

        f_adj = f(f_adj)
        b_adj = f(b_adj)

        with tf.variable_scope('Forward_pass'):
            out_fpa = message_pass(f_adj, self.fpa)
        with tf.variable_scope('Backward_pass'):
            out_bpa = message_pass(b_adj, self.bpa)

        out = tf.concat([out_fpa, out_bpa], axis=-1)
        out = tf.cast(out, tf.float32)

        return out

    def mess_build(self, G, node_order, E):
        return Messenger.build(self, G, node_order, E)
