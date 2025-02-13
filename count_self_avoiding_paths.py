import numpy as np
import torch
from torch_geometric.utils import to_dense_adj


def get_simple_paths_count(edge_index, dist_len, n_root_nodes, max_bfs_depth, max_tries_per_depth, n_reps=1,
                           sort_nodes_by_degree=False, directed=False, allow_cycles=False):

    ### Step 1 - Preliminaries
    # Adjacency
    A = to_dense_adj(edge_index)
    num_nodes = A.size(1)

    # Node degree
    deg = A[0].sum(dim=1)

    # Shortest-path distance
    a_pows = []
    Ak = A.clone().detach()
    for k in range(1, dist_len):
        a_pows.append(Ak.view(1, num_nodes * num_nodes))
        Ak = Ak @ A
    
    a_pows = torch.cat(a_pows)
    a_pows = (a_pows > 0).long()
    cs = a_pows.cumsum(dim=0).masked_fill(a_pows == 0, dist_len)
    min_val, spd = torch.min(cs, dim=0)
    spd = spd.masked_fill(min_val == dist_len, dist_len).view(num_nodes, num_nodes)

    # DAG decomposition wrapper class
    dag_dec = dag_decomposer(edge_index, num_nodes)

    ### Step 2 - Incremental SPSE matrix M construction
    sp_pows = []
    inc_degrees, tentative_nodes_ordering = deg.sort(descending=False)
    tentative_nodes_ordering = tentative_nodes_ordering.tolist()
    
    max_feasible_depth = 1 + (inc_degrees > 2).sum().item()

    for _ in range(n_reps):

        # SP matrix initialization
        m_pows = torch.zeros(dist_len, num_nodes * num_nodes, device=edge_index.device)

        nodes = tentative_nodes_ordering.copy()
        ### By default, random nodes ordering
        if not sort_nodes_by_degree:
            np.random.shuffle(nodes)
        node_hist, n = [], 0
        while n < min(num_nodes, n_root_nodes):
    
            if n == 0:
                root_node = nodes.pop(0)
                node_hist.append(root_node)
            elif n == 1:
                root_node = nodes.pop(-1)
                node_hist.append(root_node)
            else:
                rk_first = inverse_permutation(spd[node_hist[0], nodes].sort(descending=True)[1])
                rk_sec = inverse_permutation(spd[node_hist[1], nodes].sort(descending=True)[1])
                root_node = nodes.pop((rk_first + rk_sec).argsort()[0])
                node_hist.insert(0, root_node)
                node_hist.pop(-1)
            
            paths, infos = dag_dec.decompose(root_node, min(max_bfs_depth, max_feasible_depth), max_tries_per_depth)
        
            # Count n-hop directed paths between nodes and subsequently update SPSE matrix M
            for perm in paths:
                inv_perm = inverse_permutation(perm)
                permA = A[0, perm][:, perm]
                I = permA.triu(diagonal=1)
                if allow_cycles:
                    ###
                    ## Allow child nodes which are not selected during DFS / partial BFS to cycle back to the parent node for
                    ## a more thorough exploration
                    ###
                    # Mask 1: DFS / partial BFS child nodes
                    m1 = ((torch.cumsum(I, dim=1) > 1) & (torch.diff(I, dim=1, prepend=torch.zeros(num_nodes, 1)) == 1)).long()
                    m1 = torch.cumsum(m1, dim=1) * (I == 1).long()
                    # Mask 2: Nodes which are a child of a node of higher order (discovered later)
                    m2 = I * (I.flip(0).cumsum(0).flip(0) > 1).long()
                    mask = (m1 * m2).clamp(max=1)
                    # Set edge direction opposite to node ordering for nodes in the mask 
                    I = I - mask + mask.T
                i_pows = []
                Ik = I.clone()
                k = 0
                permIk = Ik[inv_perm][:, inv_perm]
                while Ik.any() and k < dist_len:
                    if directed:
                        i_pows.append(permIk.contiguous().view(1, num_nodes * num_nodes))
                    else:
                        i_pows.append((permIk.T + permIk).contiguous().view(1, num_nodes * num_nodes))
                    if allow_cycles:
                        Ik = Ik @ I
                        # Remove cycling paths
                        diag = torch.diag(Ik)
                        Ik = Ik - torch.eye(num_nodes) * diag
                        # Revert to initial node ordering
                        permIk = Ik[inv_perm][:, inv_perm]
                        # Block paths towards and from cycles' "root" nodes ==> this can be detrimental to the overall path
                        # discovery, and hence should be combined with "allow_cycles=False".
                        cycle_index = diag > 0
                        if cycle_index.any().item():
                            cycle_index = cycle_index.long()
                            # Allow only previous nodes to access a cycle's root node (i.e. can't close the cycle)
                            to_forbidden_node_mask = (torch.ones(num_nodes, num_nodes).triu(1) + \
                                        torch.ones(num_nodes, num_nodes) @ torch.diag(1 - cycle_index).float()).clamp(max=1)
                            # Make cycle's root nodes absorbing (i.e. non leavable)
                            from_forbidden_node_mask = (torch.ones(num_nodes, num_nodes).tril(-1) + \
                                    torch.diag(1 - cycle_index).float() @ torch.ones(num_nodes, num_nodes)).clamp(max=1)
                            I = I * to_forbidden_node_mask * from_forbidden_node_mask
                            ### Alternatively, the following can be used (misses a few more paths, but simpler)
                            # ones = torch.ones(num_nodes, num_nodes)
                            # I = I * (1 - (torch.diag(cycle_index).float() @ ones + ones @ torch.diag(cycle_index).float()).clamp(max=1))
                            ###
                    else:
                        Ik = Ik @ I
                        permIk = Ik[inv_perm][:, inv_perm]
                    k += 1
                # Complete with zeros
                for _ in range(k, dist_len):
                    i_pows.append(torch.zeros((1, num_nodes * num_nodes)))
                i_pows = torch.cat(i_pows)
                m_pows = torch.stack([m_pows, i_pows]).max(dim=0)[0]
                
            n += 1

        sp_pows.append(m_pows.view(dist_len, num_nodes, num_nodes))

    return torch.stack(sp_pows)


def inverse_permutation(perm):
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return inv


class dag_decomposer:


    def __init__(self, edge_index, num_nodes):
        self.edge_index = edge_index
        self.num_nodes = num_nodes


    def dbfs_step(self, root_node_idx, node_idx, visited, depth_idx, depth_switch):
    
        if node_idx not in visited:
            visited = visited + [node_idx]
        
        # Destination nodes
        child_nodes = self.edge_index[1][self.edge_index[0] == node_idx].tolist()
        child_nodes = np.array([c for c in child_nodes if c not in visited])
        np.random.shuffle(child_nodes)
        if len(child_nodes) == 0:
            self.avoid(visited)
            return visited, False
        depth_inc = 0
        if len(child_nodes) > 1:
            depth_inc = 1

        # If the required depth is reached, switch to BFS
        if depth_idx == depth_switch:
            self.avoid(visited)
            visited = self.bfs(child_nodes, visited)
            return visited, True
        
        # Else, DFS/partial-BFS
        child_nodes = child_nodes.tolist()
        retry_loop, bfs_bool, maybe_avoid = [], False, visited.copy()
        while len(child_nodes) > 0:
            child_idx = child_nodes.pop(0)
            # First check if child node has appeared in the visited list
            if (child_idx in visited):
                continue
            # If this path was previously explored, save child node for later
            if visited + [child_idx] in self.avoid_paths:
                retry_loop.insert(0, child_idx)
                continue
            # Allow a partial BFS step, that combines advantages of both DFS and BFS. This should be done at a given depth to avoid doing the
            # same decomposition over successive iterations of the decompose fn.
            if depth_idx == depth_switch - 1:
                out1, out2 = self.partial_bfs(root_node_idx, child_idx, child_nodes, visited, depth_idx + depth_inc, depth_switch)
                if out1 is None:
                    retry_loop.insert(0, child_idx)
                    continue
                visited, child_bfs_bool = out1, out2
            else:
                visited, child_bfs_bool = self.dbfs_step(root_node_idx, child_idx, visited, depth_idx + depth_inc, depth_switch)
            child_nodes.extend(retry_loop)
            retry_loop = []
            bfs_bool = True if child_bfs_bool else bfs_bool
        
        if bfs_bool is False:
            self.avoid(maybe_avoid)

        return visited, bfs_bool


    def partial_bfs(self, root_node_idx, child_idx, co_child_nodes, visited, depth_idx, depth_switch):

        unvisited_child_nodes = np.array([c for c in co_child_nodes if c not in visited])
        if len(unvisited_child_nodes) == 0:
            return self.dbfs_step(root_node_idx, child_idx, visited, depth_idx, depth_switch)
        subsets = [
            np.random.choice(unvisited_child_nodes, np.random.randint(0, len(unvisited_child_nodes)), replace=False).tolist() \
            for _ in range(5)
        ]
        subsets = [s for s in subsets if visited + [child_idx] + s not in self.avoid_paths]
        if len(subsets) == 0:
            return None, None
        bfs_nodes = subsets[0]
        if len(bfs_nodes) == 0:
            return self.dbfs_step(root_node_idx, child_idx, visited, depth_idx, depth_switch)
        else:
            self.avoid(visited + [child_idx] + bfs_nodes)
            return self.bfs(np.array([child_idx] + bfs_nodes), visited), True


    def bfs(self, child_nodes, visited):
        queue = child_nodes.tolist().copy()
        # Destination nodes
        while len(queue) > 0:
            next_idx = queue.pop(0)
            if not next_idx in visited:
                visited, queue = self.bfs_step(next_idx, visited, queue)
        return visited


    def bfs_step(self, node_idx, visited, queue):
        child_nodes = self.edge_index[1][self.edge_index[0] == node_idx].tolist()
        child_nodes = np.array([c for c in child_nodes if c not in visited])
        np.random.shuffle(child_nodes)
        return visited + [node_idx], queue + child_nodes.tolist()


    def dbfs(self, root_node_idx, depth_switch):
        visited = list()
        depth_idx = 0
        return self.dbfs_step(root_node_idx, root_node_idx, visited, depth_idx, depth_switch)[0]


    def reset_avoid_paths(self):
        setattr(self, 'avoid_paths', [])


    def avoid(self, path):
        self.avoid_paths.append(path)


    def decompose(self, root_node, max_bfs_depth, max_tries_per_depth, max_failed_tries_per_depth=5):
    
        bfs_depths = np.arange(max_bfs_depth + 1)
        paths, infos = [], []
        for depth in bfs_depths:
            self.reset_avoid_paths()
            k = depth
            kadoban = False
            failed_tries = 0
            for i in range(max_tries_per_depth):
                ordering = self.dbfs(root_node, depth)
                # Add a mechanism to keep in memory the prefix of bad DAG decompositions, 
                # and gain time by skipping these search directions
                if len(ordering) < self.num_nodes:
                    failed_tries += 1
                    if kadoban:
                        k -= 1
                    self.avoid_paths += [ordering[:k]]
                    kadoban = True
                else:
                    paths.append(ordering)
                    infos.append((root_node, depth, i))
                    kadoban = False
                if (len(ordering) == 1) or (failed_tries == max_failed_tries_per_depth) or (len(self.avoid_paths) == 0): 
                    break

        return paths, infos
