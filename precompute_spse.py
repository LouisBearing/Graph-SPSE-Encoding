import os
import argparse
import pickle
import numpy as np
import time
## Loading datasets from PyGym / custom classes
from torch_geometric.datasets import ZINC, GNNBenchmarkDataset
from PATH_TO_DATASET_DIRECTORY.peptides_functional import \
    PeptidesFunctionalDataset # These two require a `Dataset` implementation (as opposed to `InMemoryDataset`) 
from PATH_TO_DATASET_DIRECTORY.peptides_structural import \
    PeptidesStructuralDataset #
from count_self_avoiding_paths import get_simple_paths_count
from ogb.lsc import PygPCQM4Mv2Dataset


def main(args):

    t0 = time.time()
    out_dirp = args.datasetp
    splits = ['train', 'val', 'test'] if args.split == 'all' else [args.split]
    for split in splits:
        split_count = 0
        if (args.dataset_name == 'CLUSTER') or (args.dataset_name == 'PATTERN') or (args.dataset_name == 'MNIST') \
            or (args.dataset_name == 'CIFAR10'):
            ds = GNNBenchmarkDataset(root=args.datasetp, name=args.dataset_name, split=split)
            out_dirp = os.path.join(args.datasetp, f'{args.dataset_name}_SP_mat')
            os.makedirs(out_dirp, exist_ok=True)
        elif args.dataset_name == 'ZINC':
            ds = ZINC(root=args.datasetp, subset=True, split=split)
        elif (args.dataset_name == 'PeptidesFunctional'):
            ds = PeptidesFunctionalDataset(args.datasetp)
            out_dirp = os.path.join(args.datasetp, 'PeptidesFunc_SP_mat')
            os.makedirs(out_dirp, exist_ok=True)
        elif (args.dataset_name == 'PeptidesStructural'):
            ds = PeptidesStructuralDataset(args.datasetp)
        elif (args.dataset_name == 'PCQM4Mv2'):
            ds = PygPCQM4Mv2Dataset(root=args.datasetp)
            out_dirp = os.path.join(args.datasetp, 'PCQM4Mv2_SP_mat')
            os.makedirs(out_dirp, exist_ok=True)
        sp_mat = []
        write_bool = False
        for i in range(len(ds)):
            write_p = os.path.join(out_dirp, f'SP_mat+{args.dataset_name}_{split}_{split_count}')
            if write_bool or (not os.path.exists(write_p)):
                write_bool = True
                # Placeholder
                if not os.path.exists(write_p):
                    print(f'Placeholder for split {write_p}')
                    with open(write_p, 'wb') as f:
                        pickle.dump(list(), f)
                spl = ds[i]
                n_root_nodes = max(8, int(np.round(len(spl.x) * args.n_root_nodes_fact)))
                sp_mat.append(get_simple_paths_count(spl.edge_index, dist_len=args.dist_len, 
                                                        n_root_nodes=n_root_nodes, 
                                                        max_bfs_depth=args.max_bfs_depth, 
                                                        max_tries_per_depth=args.max_tries_per_depth, 
                                                        n_reps=args.n_reps, directed=args.directed,
                                                        allow_cycles=args.cycles).max(dim=0, keepdim=True)[0])
            if i % 100 == 0:
                print(f'iter {i}, time: {time.time() - t0}')
            ## Saving path count matrix in different files to preserve memory
            if (i > 0) and \
                (
                    ((args.dataset_name in ['CLUSTER', 'PeptidesFunctional']) and (i % 2000 == 0))
                    or
                    ((args.dataset_name == 'PATTERN') and (i % 4000 == 0))
                    or
                    ((args.dataset_name == 'PCQM4Mv2') and (i % 200000 == 0))
                    or
                    ((args.dataset_name  in ['CIFAR10', 'MNIST']) and (i % 10000 == 0))
                ):
                if write_bool:
                    with open(write_p, 'wb') as f:
                        pickle.dump(sp_mat, f)
                split_count += 1
                write_bool = False
                sp_mat = []
        if write_bool:
            with open(write_p, 'wb') as f:
                pickle.dump(sp_mat, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetp', default=r'datasets', help='Path to I/O directory')
    parser.add_argument('--dataset_name', help='Which dataset')
    parser.add_argument('--n_root_nodes_fact', default=0.17, type=float,
                        help="Proportion of all nodes to be used as initial node for the search.")
    parser.add_argument('--max_bfs_depth', default=3, type=int,
                        help="Maximum depth for switching from DFS to BFS (+1 wrt paper notations).")
    parser.add_argument('--max_tries_per_depth', default=6, type=int,
                        help="Number of trials for each per value of the preceding parameter to account for random effects.")
    parser.add_argument('--dist_len', default=8, type=int,
                        help="Maximum search depth, set by the dataset configuration.")
    parser.add_argument('--n_reps', default=1, type=int,
                        help="Number if iterations of the overall algorithm.")
    parser.add_argument('--split', default='all', type=str)
    parser.add_argument('--directed', action='store_true', default=False,
                    help="By default, allows a more efficient simple path discovery for undirected graphs.")
    parser.add_argument('--cycles', action='store_true', default=False,
                    help="Cycles are forbidden by default, and can be allowed by setting this to true.")
    args = parser.parse_args()
    print('Arguments received, calling main !')
    main(args)

