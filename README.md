# Graph-SPSE-Encoding
The minimal reproducible code to count simple paths in a `PyTorch Geometric` graph in Python.

### What it is

This is the repository for our ICML 2025 paper: "Simple Path Structural Encoding for Graph Transformers".

SPSE can be used by replacing random walk probabilities with simple path counts in transformer models such as [GRIT](https://github.com/LiamMa/GRIT), [CSA](https://github.com/inria-thoth/csa) or [Graph-GPS](https://github.com/rampasek/GraphGPS).

For low-density/short path lengths, exact path counting methods exist (check NetworkX's `all_simple_paths` function: [NetworkX documentation](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.simple_paths.all_simple_paths.html).

### What it contains

This repository contains the implementation of a proposed DFS/BFS-based approximate counting algorithm.

``count_self_avoiding_paths.py`` contains the function ``get_simple_paths_count`` which counts simple paths given the edge index matrix of a `PyTorch Geometric` input graph.

The second file, ``precompute_spse.py``, describes the overall path counting process for standard PyG databases. It notably involves storing the path counts into several, smaller size files.
