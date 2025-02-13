# Graph-SPSE-Encoding
The minimal reproducible code to count simple paths in a `PyTorch Geometric` graph in python.

``count_self_avoiding_paths.py`` contains the function ``get_simple_paths_count`` which count simple paths given the edge index matrix of a `PyTorch Geometric` input graph.

The second file, ``precompute_spse.py``, describes the overall path counting process for standard PyG databases. It notably involves storing the path counts into several, smaller size files.
