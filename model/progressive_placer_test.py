import os
import networkx as nx
import numpy as np
import tensorflow as tf
import sys
import pickle

from numpy.lib._iotools import str2bool

sys.path.append('./')
sys.path.append('progressive_placers/')
sys.path.append('sim/')
sys.path.append('model/')
from model.progressive_placer import *

import argparse
import model.rl_params
from itertools import chain
from model.mp_progressive_nn import MessagePassingProgressiveNN
from model.simple_nn import *
from model.simple_graphs import *
from model.pp_item import *


# from utils import *


class ProgressivePlacerTest(object):

    @staticmethod
    def sim_reward(n_devs, sim, p, G):
        # Penalty for one unit of outshape transfer
        l_fact = 1

        costs = nx.get_node_attributes(G, 'cost')
        op_memories = nx.get_node_attributes(G, 'out_size')

        run_time, dft, du, tom, nt, start_times = sim.get_runtime(G, n_devs,
                                                                  costs, op_memories, p)

        assert tom >= 0
        run_time += l_fact * tom

        return run_time, start_times

    '''
    place everything on the last gpu (id: n_devs-1)
    '''

    @staticmethod
    def sim_single_gpu(n_devs, sim, p, G):
        start_times = {}
        for n in G.nodes():
            start_times[n] = 0.
        run_time = 0
        for _, d in p.items():
            if d != n_devs - 1:
                run_time += 1

        return run_time, start_times

    @staticmethod
    def sim_neigh_placement(n_devs, sim, p, G):
        start_times = {}
        for n in G.nodes():
            start_times[n] = 0.
        run_time = 0
        for n, d in p.items():
            for neigh in chain(G.neighbors(n), G.predecessors(n)):
                if p[neigh] != p[n]:
                    run_time += 1

        return run_time, start_times, [0] * 2

    @staticmethod
    def choose_model(model_name):

        # if model_name == 'supervised':
        #   nn_model = SupervisedSimpleNN
        if model_name == 'simple_nn':
            nn_model = SimpleNN
        # elif model_name == 'local_nn':
        #   nn_model = LocalProgressiveNN
        elif model_name == 'mp_nn':
            nn_model = MessagePassingProgressiveNN
        # elif model_name == 'or':
        #   nn_model = MessagePassingOneRewardNN
        else:
            raise Exception('%s not implemented model' % model_name)

        return nn_model

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
            return False

    def test(self, config):
        graph = config['graph']
        N = config['graph_size']
        n_devices = config['n_devs']
        m_name = config['m_name']
        f = None
        sim = None
        # if graph in ['chain', 'crown', 'edge']:
        #   from old_simulator import LegacySimulator
        #   sim = LegacySimulator(None, False, n_devs=n_devs, override_ban=True)

        # TODO 1
        # create a graph - either syhthetic, or from a file
        if graph == 'chain':
            G = makeChainGraph(N, n_devices)
        elif graph == 'crown':
            G = makeCrownGraph(N, n_devices)
        elif graph == 'edge':
            G = makeEdgeGraph(N)
        else:
            input_file = config['pickled_inp_file'][0]
            if config['local_prefix'] is not None:
                input_file = config['local_prefix'] + '/' + config['pickled_inp_file'][0]

            pptf = PPTFItem(input_file, n_devices,
                            simplify_tf_reward_model=config['simplify_tf_rew_model'],
                            use_new_sim=config['use_new_sim'],
                            sim_mem_usage=True,
                            # sim_mem_usage=False,
                            final_size=config['prune_final_size'])

            if not config['simplify_tf_rew_model']:
                f = lambda _, __, p, ___: pptf.simulate(p)
            G = pptf.get_grouped_graph()

        Gg = G
        pos = nx.spiral_layout(Gg)

        from itertools import count
        groups = set(range(config['n_devs']))
        nodesToPrint = Gg.nodes
        # mappingToPrint = dict(zip(sorted(groups), count()))
        # colorsToPrint = [mappingToPrint[Gg.nodes[n]['placement']] for n in nodesToPrint]

        # https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html for CMAP
        nc = nx.draw_networkx_nodes(Gg, pos,
                                    nodelist=nodesToPrint,
                                    # node_color=colorsToPrint,
                                    with_labels=True,
                                    node_size=10,
                                    cmap='plasma')

        # TODO remove for remote play
        plt.colorbar(nc)
        plt.axis('off')
        nx.draw_networkx_edges(Gg, pos, width=1.0, alpha=0.2)
        # placement_graph_img_location = '%s/graph-%d.png' % (self.fig_dir, ep)
        # plt.savefig(placement_graph_img_location, dpi=300)
        plt.show()
        plt.clf()

        if not f:
            if config['rew_singlegpu']:
                f = ProgressivePlacerTest.sim_single_gpu
            elif config['rew_neigh_pl']:
                f = ProgressivePlacerTest.sim_neigh_placement
            else:
                f = ProgressivePlacerTest.sim_reward

        if config['eval'] is not None:
            _, r, ss, p = pptf.eval(config['eval'])

            fname = 'models/chrome-traces/%s/timeline.json' % (config['name'])
            # timeline_to_json(ss, p, fname)
        else:
            ProgressivePlacer().place(
                G, n_devices, ProgressivePlacerTest.choose_model(m_name),
                lambda *args, **kwargs: f(n_devices, sim, *args, **kwargs),
                config)
            # ProgressivePlacer().place(
            #     G, n_devices, ProgressivePlacerTest.choose_model(m_name),
            #     lambda *args, **kwargs: f(n_devices, sim, *args, **kwargs),
            #     config, pptf)

    def mul_graphs(self, config):
        from model.coord import Coordinator
        Coordinator().start(config, self.test)

    # def benchmark_policy(self, config):
    # from model.policy_benchmarker import PolicyBenchmarker
    # PolicyBenchmarker().start(config, self.test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--name', '-n', type=str, default='test')
    parser.add_argument('--graph', '-g', type=str, default=None)
    parser.add_argument('--id', type=int, default=None)
    # for synthetic non-tensorflow graphs
    parser.add_argument('--graph-size', '-N', type=int, default=4)
    parser.add_argument('--pickled-inp-file', '-i', type=str, default=None, nargs='+')
    parser.add_argument('--mul-graphs', type=str, default=None, nargs='+')
    parser.add_argument('--dataset-folder', '-dataset', type=str, default=None,
                        help='Use this to denote a folder containing a dataset like cifar10. '
                             'Each subfolder will be checked for input.pkl files')

    parser.add_argument('--n-devs', type=int, default=2)
    parser.add_argument('--model-folder-prefix', type=str, default='', dest='model_folder_prefix')

    # progressive placer model args
    parser.add_argument('--m-name', type=str, default='mp_nn')
    parser.add_argument('--n-peers', type=int, default=None)
    parser.add_argument('--agg-msgs', type=str2bool, dest='agg_msgs')
    parser.add_argument('--no-msg-passing', type=str2bool, dest='no_msg_passing')
    parser.add_argument('--radial-mp', type=int, default=None)
    parser.add_argument('--tri-agg', type=str2bool, dest='tri_agg')
    parser.add_argument('--sage', type=str2bool, dest='sage')
    parser.add_argument('--sage-hops', type=int, dest='sage_hops')
    parser.add_argument('--sage-sample-ratio', type=float, dest='sage_sample_ratio')
    parser.add_argument('--sage-dropout-rate', type=float, dest='sage_dropout_rate')
    parser.add_argument('--sage-aggregation', type=str, dest='sage_aggregation', default='mean')

    parser.add_argument('--sage-position-aware', type=str2bool, dest='sage_position_aware')
    parser.add_argument('--use-single-layer-perceptron', type=str2bool, dest='use_single_layer_perceptron')
    parser.add_argument('--pgnn-c', type=float, dest='pgnn_c')
    parser.add_argument('--pgnn-neigh-cutoff', type=int, dest='pgnn_neigh_cutoff')
    parser.add_argument('--pgnn-anchor-exponent', type=int, dest='pgnn_anchor_exponent')
    parser.add_argument('--pgnn-aggregation', type=str, dest='pgnn_aggregation', default='max')
    parser.add_argument('--reinit-model', type=str2bool, dest='reinit_model')

    # training args
    parser.add_argument('--n-eps', type=int, default=int(1e9))
    parser.add_argument('--max-rnds', type=int, default=None)
    parser.add_argument('--disc-factor', type=float, default=1.)
    parser.add_argument('--vary-init-state', dest='vary_init_state', action='store_true')
    parser.add_argument('--zero-placement-init', dest='zero_placement_init', action='store_true')
    parser.add_argument('--null-placement-init', dest='null_placement_init', action='store_true')
    parser.add_argument('--init-best-pl', dest='init_best_pl', action='store_true')
    parser.add_argument('--one-shot-episodic-rew', dest='one_shot_episodic_rew', action='store_true')
    parser.add_argument('--ep-decay-start', type=float, default=1e3)
    parser.add_argument('--bl-n-rnds', type=int, default=1000)
    parser.add_argument('--rew-singlegpu', dest='rew_singlegpu', action='store_true')
    parser.add_argument('--rew-neigh-pl', dest='rew_neigh_pl', action='store_true')
    parser.add_argument('--supervised', dest='supervised', action='store_true')
    parser.add_argument('--use-min-runtime', dest='use_min_runtime', action='store_true')
    parser.add_argument('--discard-last-rnds', dest='discard_last_rnds', action='store_true')
    parser.add_argument('--turn-based-baseline', dest='turn_based_baseline', action='store_true')
    parser.add_argument('--dont-repeat-ff', action='store_true', dest='dont_repeat_ff')
    parser.add_argument('--small-nn', action='store_true', dest='small_nn')
    parser.add_argument('--dont-restore-softmax', dest='dont_restore_softmax', action='store_true')
    parser.add_argument('--restore-from', type=str, default=None)

    # report/log args
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('--save-freq', type=int, default=100)
    parser.add_argument('--eval-freq', type=int, default=999)
    parser.add_argument('--log-tb-workers', dest='log_tb_workers', action='store_true')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--debug-verbose', dest='debug_verbose', action='store_true')
    parser.add_argument('--disamb-pl', dest='disamb_pl', action='store_true')
    parser.add_argument('--eval', type=str, default=None)
    parser.add_argument('--simplify-tf-rew-model', action='store_true', dest='simplify_tf_rew_model')
    parser.add_argument('--log-runtime', dest='log_runtime', action='store_true')
    parser.add_argument('--use-new-sim', action='store_true', dest='use_new_sim')
    parser.add_argument('--gen-profile-timeline', dest='gen_profile_timeline', action='store_true')
    parser.add_argument('--mem-penalty', type=float, default=0.)
    parser.add_argument('--max-mem', type=float, default=11., help='Default Max Memory of GPU (in GB)')
    parser.add_argument('--max-runtime-mem-penalized', type=float, default=10.,
                        help='Instantaneous runtime of the placement after adding the memory penalty has to be lower than this number. Note that improvement in this memory penalized runtime metric is used to compute intermediate rewards')

    # dist training params
    parser.add_argument('--use-threads', dest='use_threads', action='store_true')
    parser.add_argument('--scale-norm', dest='scale_norm', action='store_true')
    parser.add_argument('--dont-share-classifier', action='store_true', dest='dont_share_classifier')
    parser.add_argument('--use-gpus', type=str, nargs='+', default=None)
    parser.add_argument('--eval-on-transfer', type=int, default=None,
                        help='Number of episodes to transfer train before reporting eval runtime')
    parser.add_argument('--normalize-aggs', dest='normalize_aggs', action='store_true')
    parser.add_argument('--bn-pre-classifier', dest='bn_pre_classifier', action='store_true')
    parser.add_argument('--bs', type=int, default=None)
    parser.add_argument('--num-children', type=int, default=1)
    parser.add_argument('--disable-profiling', action='store_true', dest='disable_profiling')
    parser.add_argument('--n-async-sims', type=int, default=None)
    parser.add_argument('--baseline-mask', type=int, nargs='+', default=None)
    parser.add_argument('--n-workers', type=int, default=1)
    parser.add_argument('--node-traversal-order', default='topo', help='Options: topo, random')
    parser.add_argument('--prune-final-size', type=int, default=None)
    parser.add_argument('--dont-sim-mem', dest='dont_sim_mem', action='store_true')

    parser.add_argument('--remote-async-addrs', type=str, default=None, nargs='+')
    parser.add_argument('--remote-async-start-ports', type=int, default=None, nargs='+')
    parser.add_argument('--remote-async-n-sims', type=int, default=None, nargs='+')
    parser.add_argument('--local-prefix', type=str, default=None)
    parser.add_argument('--remote-prefix', type=str, default=None)
    parser.add_argument('--shuffle-gpu-order', dest='shuffle_gpu_order', action='store_true')

    args, unknown = parser.parse_known_args()

    # assert args.dont_repeat_ff

    if args.one_shot_episodic_rew and args.n_async_sims is not None:
        raise Exception('Input setting leads to deadlock')

    if args.eval_freq % 10 == 0:
        print('Eval freq cannot be divisible by 10')
        sys.exit(0)

    for option in unknown:
        for i in range(len(option)):
            if option[i] != '-':
                break
        if i > 0:
            option = option[i:].replace('-', '_')
            if option not in model.rl_params.args.__dict__:
                print(option)
                # pass
                raise Exception("Passed unknown option in dict : %s" % option)

    if args.use_gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ' '.join(args.use_gpus)

    # if args.eval_on_transfer is not None:
    # ProgressivePlacerTest().benchmark_policy(args.__dict__)
    if args.n_workers > 1:
        ProgressivePlacerTest().mul_graphs(args.__dict__)
    else:
        start_time = time()
        if(args.__dict__['dataset_folder'] is not None):
            r = []
            for root, dirs, files in os.walk(args.__dict__['dataset_folder']):
                for name in files:
                    args.__dict__['dataset'] = root.split("/")[-1]
                    args.__dict__['pickled_inp_file'] = [os.path.join(root, name)]
                    ProgressivePlacerTest().test(args.__dict__)
                    # r.append(os.path.join(root, name))

        ProgressivePlacerTest().test(args.__dict__)
        print("Test time (minutes): ", (time() - start_time) / 60)
