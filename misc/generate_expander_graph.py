import math
import numpy as np
import scipy as sp
from typing import Any, Optional
import torch

def laplacian_matrix(senders: np.ndarray, receivers: np.ndarray,
        weights: Optional[np.ndarray] = None, n: Optional[int] = None) -> Any:
  if weights is None:
    weights = 0*senders + 1

  if n is None:
    n = senders.max()
    if receivers.max() > n:
      n = receivers.max()
    n += 1

  s = senders.tolist() + list(range(n))
  t = receivers.tolist() + list(range(n))
  w = weights.tolist() + [0.0] * n
  adj = sp.sparse.csc_matrix((w, (s, t)), shape=(n, n))
  lap = adj * -1.0
  lap.setdiag(np.ravel(adj.sum(axis=0)))
  return lap


def laplacian_eigenv(senders: np.ndarray,
                     receivers: np.ndarray,
                     weights: Optional[np.ndarray] = None,
                     k=2,
                     n: Optional[int] = None):

    m = senders.shape[0]
    if weights is None:
        weights = np.ones(m)

    if n is None:
      n = senders.max()
      if receivers.max() > n:
        n = receivers.max()
      n += 1

    lap_mat = laplacian_matrix(senders, receivers, weights, n = n)
    k = min(n - 2, k + 1)
  
    eigenvals, eigenvecs = sp.sparse.linalg.eigs(lap_mat, k=k, which='SM')
    eigenvals = np.real(eigenvals)
    eigenvecs = np.real(eigenvecs)

    sorted_idx = np.argsort(eigenvals)
    eigenvals = eigenvals[sorted_idx]
    eigenvecs = eigenvecs[:, sorted_idx]

    constant_eigenvec_idx = 0

    for i in range(0, k):
      
        eigenvecs[:, i] = eigenvecs[:, i] / np.sqrt((eigenvecs[:, i]**2).sum())
        if eigenvecs[:, i].var() <= 1e-7:
            constant_eigenvec_idx = i

    non_constant_idx = [*range(0, k)]
    non_constant_idx.remove(constant_eigenvec_idx)

    eigenvals = eigenvals[non_constant_idx]
    eigenvecs = eigenvecs[:, non_constant_idx]

    return eigenvals, eigenvecs

def generate_random_regular_graph1(num_nodes, degree, rng=None):

  if rng is None:
    rng = np.random.default_rng()

  senders = [*range(0, num_nodes)] * degree
  receivers = []
  for _ in range(degree):
    receivers.extend(rng.permutation(list(range(num_nodes))).tolist())

  senders, receivers = [*senders, *receivers], [*receivers, *senders]

  senders = np.array(senders)
  receivers = np.array(receivers)

  return senders, receivers



def generate_random_regular_graph2(num_nodes, degree, rng=None):

  if rng is None:
    rng = np.random.default_rng()

  senders = [*range(0, num_nodes)] * degree
  receivers = rng.permutation(senders).tolist()

  senders, receivers = [*senders, *receivers], [*receivers, *senders]

  return senders, receivers


def generate_random_graph_with_hamiltonian_cycles(num_nodes, degree, rng=None):

  if rng is None:
    rng = np.random.default_rng()

  senders = []
  receivers = []
  for _ in range(degree):
    permutation = rng.permutation(list(range(num_nodes))).tolist()
    for idx, v in enumerate(permutation):
      u = permutation[idx - 1]
      senders.extend([v, u])
      receivers.extend([u, v])

  senders = np.array(senders)
  receivers = np.array(receivers)

  return senders, receivers


def generate_random_expander(num_nodes, degree, algorithm, rng=None, max_num_iters=100, exp_index=0):

  if rng is None:
    rng = np.random.default_rng()

  eig_val = -1
  eig_val_lower_bound = max(0, 2 * degree - 2 * math.sqrt(2 * degree - 1) - 0.1)

  max_eig_val_so_far = -1
  max_senders = []
  max_receivers = []
  cur_iter = 1

  if num_nodes <= degree:
    degree = num_nodes - 1

  if num_nodes <= 10:
    for i in range(num_nodes):
      for j in range(num_nodes):
        if i != j:
          max_senders.append(i)
          max_receivers.append(j)
  else:
    while eig_val < eig_val_lower_bound and cur_iter <= max_num_iters:
      if algorithm == 'Random-d':
        senders, receivers = generate_random_regular_graph1(num_nodes, degree, rng)
      elif algorithm == 'Random-d-2':
        senders, receivers = generate_random_regular_graph2(num_nodes, degree, rng)
      elif algorithm == 'Hamiltonian':
        senders, receivers = generate_random_graph_with_hamiltonian_cycles(num_nodes, degree, rng)
      [eig_val, _] = laplacian_eigenv(senders, receivers, k=1, n=num_nodes)
      if len(eig_val) == 0:
        print("num_nodes = %d, degree = %d, cur_iter = %d, mmax_iters = %d, senders = %d, receivers = %d" %(num_nodes, degree, cur_iter, max_num_iters, len(senders), len(receivers)))
        eig_val = 0
      else:
        eig_val = eig_val[0]

      if eig_val > max_eig_val_so_far:
        max_eig_val_so_far = eig_val
        max_senders = senders
        max_receivers = receivers

      cur_iter += 1

  non_loops = [*filter(lambda i: max_senders[i] != max_receivers[i], range(0, len(max_senders)))]

  senders = np.array(max_senders)[non_loops]
  receivers = np.array(max_receivers)[non_loops]

  max_senders = torch.tensor(max_senders, dtype=torch.long).view(-1, 1)
  max_receivers = torch.tensor(max_receivers, dtype=torch.long).view(-1, 1)

  expander_edge_index = torch.cat([max_senders, max_receivers], dim=1)

  expander_edge_index = expander_edge_index.T

  expander_edge_attr = torch.ones((expander_edge_index.size(1), 1))

  return expander_edge_index, expander_edge_attr


def create_virtual_graph(num_nodes, num_features, num_virtual_nodes):
    if num_virtual_nodes <= 0:
        return torch.empty(0, num_features), torch.empty(2, 0, dtype=torch.long), torch.empty(0, 1)

    virtual_nodes = torch.zeros((num_virtual_nodes, num_features))  
    source_indices = torch.arange(num_nodes).repeat_interleave(num_virtual_nodes)
    target_indices = torch.arange(num_virtual_nodes).repeat(num_nodes)
    virtual_edge_index = torch.stack([source_indices, target_indices], dim=0)

    num_virtual_edges = num_nodes * num_virtual_nodes
    virtual_edge_attributes = torch.ones((num_virtual_edges, 1))

    return virtual_nodes, virtual_edge_index, virtual_edge_attributes

num_layers = 5
algorithm = 'Random-d'
num_virtual_nodes = 5
nxyz = 64
num_nodes = (nxyz + 2)**3
num_features = 64
degree = 10

exphormer_chars = []

for i in range(num_layers):

    rng = np.random.default_rng(seed = 42 + i)

    expander_edge_index, expander_edge_attr = generate_random_expander(num_nodes, degree, algorithm, rng = rng, max_num_iters=100, exp_index=0)

    virtual_nodes, virtual_edge_index, virtual_edge_attr = create_virtual_graph(num_nodes, num_features, num_virtual_nodes)

    expander_edge_index = torch.tensor(expander_edge_index, dtype=torch.long)
    expander_edge_attr = torch.tensor(expander_edge_attr, dtype=torch.float32)

    if num_virtual_nodes != 0:

        virtual_nodes = torch.tensor(virtual_nodes, dtype=torch.float32) 
        virtual_edge_index = torch.tensor(virtual_edge_index, dtype=torch.long)  
        virtual_edge_attr = torch.tensor(virtual_edge_attr, dtype=torch.float32)

    else:
        virtual_nodes = 0
        virtual_edge_index = 0
        virtual_edge_attr = 0

    exphormer_chars.append([expander_edge_index, expander_edge_attr, virtual_nodes, virtual_edge_index, virtual_edge_attr])


torch.save(exphormer_chars, '/content/drive/My Drive/Project 3D/data/expander_graphs/num_layers_' + str(num_layers) + '_expdegree_' + str(degree) + '_expalgorithm_' + algorithm + '_numvirtnodes_' + str(num_virtual_nodes) + '_hidden_channels_' + str(num_features) + '_nxyz_' + str(nxyz) + '.pth')

