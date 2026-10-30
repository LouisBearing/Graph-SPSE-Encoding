# Graph-SPSE-Encoding
The minimal reproducible code to count simple paths in a `PyTorch Geometric` graph in Python.

### What it is

This is the repository for our ICML 2025 paper: "Simple Path Structural Encoding for Graph Transformers".

SPSE can be used by replacing random walk probabilities with simple path counts in transformer models such as [GRIT](https://github.com/LiamMa/GRIT), [CSA](https://github.com/inria-thoth/csa) or [Graph-GPS](https://github.com/rampasek/GraphGPS).

### Exact SPSE calculation on low-density / small graphs

Most of the time, an exact path counting approach, e.g. using iGraph, suffices (check also NetworkX's `all_simple_paths` function on [NetworkX documentation](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.simple_paths.all_simple_paths.html)). See for instance how to compute SPSE on ZINC below.

```python
import torch
import os.path as osp
import numpy as np
import igraph as ig
from torch_geometric.datasets import ZINC
import inspect

def get_simple_path_count_igraph(graph, path_length):
    '''
    Expects a igraph input graph.
    '''
    
    kwargs = (
        {"cutoff": path_length}
        if "cutoff" in inspect.signature(graph.get_all_simple_paths).parameters
        else {"maxlen": path_length}
    )

    nnodes = graph.vcount()
    path_count_matrix = np.zeros((nnodes, nnodes, path_length))
    
    for i in range(nnodes):
        for j in range(i):

            path_count = np.zeros(path_length, dtype=np.int64)

            for path in graph.get_all_simple_paths(j, to=i, **kwargs): # maxlen for legacy, cutoff in newer api version
                path_count[len(path) - 2] += 1

            path_count_matrix[i, j] = path_count
            path_count_matrix[j, i] = path_count
            
            j += 1

    return torch.from_numpy(path_count_matrix)


pl = 8 ## Define maximum path length
data_dir = '' # PATH TO DATA
save, save_dir = False, '' # WHERE TO SAVE SPSE
dataset = ZINC(root=osp.join(data_dir, 'ZINC'), subset=True, split='train')

for i in range(len(dataset)):
    g = dataset[i]
    nnodes = g.num_nodes
    # Map to igraph
    ig_graph = ig.Graph(n=nnodes, edges=g.edge_index.T.tolist(), directed=True)
    # SPSE calculation
    spse = get_simple_path_count_igraph(ig_graph, path_length=pl) # SPSE shape: nnodes x nnodes x pl
    # Maybe save results
    if save:
        filename = osp.join(save_dir, f'zinc_spse_{i}_len{pl}.pt')
        torch.save(spse.to(torch.float32), filename)
```

### What this repository contains

This repository contains the implementation of a proposed DFS/BFS-based approximate path counting algorithm, for cases where the exact counting approach is unfeasible.

``count_self_avoiding_paths.py`` contains the function ``get_simple_paths_count`` which counts simple paths given the edge index matrix of a `PyTorch Geometric` input graph.

The second file, ``precompute_spse.py``, describes the overall path counting process for standard PyG databases. It notably involves storing the path counts into several, smaller size files.


## Citation

```
@inproceedings{airale2025simple,
  title={Simple Path Structural Encoding for Graph Transformers},
  author={Airale, Louis and Longa, Antonio and Rigon, Mattia and Passerini, Andrea and Passerone, Roberto},
  booktitle={International Conference on Machine Learning},
  pages={857--873},
  year={2025},
  organization={PMLR}
}
```
