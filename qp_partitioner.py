import os
import random
from collections import defaultdict
import multiprocessing as mp

import pandas as pd
pd.options.display.float_format = '{:,.8f}'.format
import matplotlib.pyplot as plt
from scipy.linalg import solve


def area(region):
  x1, x2, y1, y2 = region
  return (x2 - x1) * (y2 - y1)


def split_region(rgn, type_='v'):
  x1, x2, y1, y2 = rgn
  if type_ not in ('v', 'h'):
    raise ValueError(f"Type must be v or h, and not {type_}")

  if type_ == 'v':
    rgns = [
      (x1, (x1 + x2)/2, y1, y2),
      ((x1 + x2)/2, x2, y1, y2)
    ]
  else:
    rgns = [
      (x1, x2, y1, (y1 + y2)/2),
      (x1, x2, (y1 + y2)/2, y2)
    ]
  return rgns


def propagate(to_x1, to_x2, to_y1, to_y2, from_x, from_y):
  if from_x < to_x1:
    from_x = to_x1
  elif from_x > to_x2:
    from_x = to_x2

  if from_y < to_y1:
    from_y = to_y1
  elif from_y > to_y2:
    from_y = to_y2
  return from_x, from_y


class Graph:
  def __init__(self, edges, pads, X, Y):
    self.edges = edges
    self.X = X
    self.Y = Y

    self.if_pad = None
    self.set_if_pad(pads)

  def set_if_pad(self, pads):
    self.if_pad = {}
    for i in self.edges.index:
      if i in pads:
        self.if_pad[i] = 1
      else:
        self.if_pad[i] = 0
    self.if_pad = pd.Series(self.if_pad).sort_index()

  def bft(self, start_node, visited):
    # assign default visited as [0] * num_nodes
    # visited is to be passed by reference
    nodes = []
    for j in self.edges.columns:
      if self.edges[start_node][j] != 0 and visited[j] == 0:
        nodes.append(j)

    for node in nodes:
      visited[node] = 1

    for node in nodes:
      self.bft(node, visited)

  def dft(self, start_node, visited):
    # assign default visited as [0] * num_nodes
    # visited is to be passed by reference
    nodes = []
    for j in self.edges.columns:
      if self.edges[start_node][j] != 0 and visited[j] == 0:
        nodes.append(j)

    for node in nodes:
      visited[node] = 1
      self.dft(node, visited)

  def qp(self):
    gates = sorted(self.if_pad[self.if_pad == 0].index)
    pads = sorted(self.if_pad[self.if_pad == 1].index)

    C = self.edges.loc[gates, gates]

    A = -(C.copy())
    for i in A.index:
      for j in A.columns:
        if i == j:
          A.loc[i, i] = self.edges.loc[i].sum()

    bx = {gate: self.edges.loc[gate, pads].dot(self.X[pads]) for gate in gates}
    by = {gate: self.edges.loc[gate, pads].dot(self.Y[pads]) for gate in gates}
    bx, by = pd.Series(bx).sort_index(), pd.Series(by).sort_index()

    X = solve(A, bx)
    X = pd.Series({gates[i]: X[i] for i in range(len(gates))})
    for pad in pads:
      X[pad] = self.X[pad]
    self.X = X

    Y = solve(A, by)
    Y = pd.Series({gates[i]: Y[i] for i in range(len(gates))})
    for pad in pads:
      Y[pad] = self.Y[pad]
    self.Y = Y

  @staticmethod
  def split_sorted_gates(gates):
    half_1 = gates[:int(len(gates)/2)]
    half_2 = gates[int(len(gates)/2):]
    return half_1, half_2

  def _partition(self, gates_in_region, gates_out_region, region_to_shift_to):
    if not len(gates_in_region):
      return

    pads = list(self.if_pad[self.if_pad == 1].index)
    edges = self.edges.copy()

    # get all connected gates and pads to first half of the gates 
    # (which will include gates_in_region gates)
    connected_nodes = []
    for node in gates_in_region:
      connected_nodes.extend(list(edges[node][edges[node] != 0].index))
    all_nodes = list(set(connected_nodes + gates_in_region))
    edges = edges.loc[all_nodes, all_nodes]
    X = self.X.loc[all_nodes]
    Y = self.Y.loc[all_nodes]

    # shift connected pads to partition line
    connected_pads = list(set(connected_nodes) & set(gates_out_region + pads))
    x1, x2, y1, y2 = region_to_shift_to
    for pad in connected_pads:
      X[pad], Y[pad] = propagate(x1, x2, y1, y2, X[pad], Y[pad])

    # create a graph with same edges, and additional pads from gates_out_region gates,
    # updated locations of pads shifted to partition line
    sub_graph = Graph(edges, connected_pads, X, Y)
    sub_graph.qp()
    self.X[gates_in_region] = sub_graph.X.loc[gates_in_region]
    self.Y[gates_in_region] = sub_graph.Y.loc[gates_in_region]

  def partition_4(self, gates_to_split, region):
    if len(gates_to_split) == 0:
      return

    if area(region) < 160:
      return

    gates = self.if_pad[self.if_pad == 0].index
    gates = list(pd.concat([self.X.loc[gates], self.Y.loc[gates]],
      axis=1).apply(tuple, axis=1).sort_values().index)

    halfs = self.split_sorted_gates(gates_to_split)
    regions = split_region(region, type_='v')
    self._partition(halfs[0], [gate for gate in gates if gate not in halfs[0]], regions[0])
    self._partition(halfs[1], [gate for gate in gates if gate not in halfs[1]], regions[1])

    regions_i, halfs_i = [], []
    for i in range(2):
      regions_i.append(split_region(regions[i], type_='h'))
      halfs_i.append(self.split_sorted_gates(halfs[i]))
      self._partition(halfs_i[i][0], [gate for gate in gates if gate not in halfs_i[i][0]], regions_i[i][0])
      self._partition(halfs_i[i][1], [gate for gate in gates if gate not in halfs_i[i][1]], regions_i[i][1])

    for i in range(2):
      for j in range(2):
        self.partition_4(halfs_i[i][j], regions_i[i][j])

  def partition(self):
    gates = self.if_pad[self.if_pad == 0].index
    gates = list(pd.concat([self.X.loc[gates], self.Y.loc[gates]],
      axis=1).apply(tuple, axis=1).sort_values().index)

    self.partition_4(gates, (0, 100, 0, 100))

  def plot(self):
    mx_weight = self.edges.max().max()
    for i in self.edges.index:
      for j in self.edges.columns:
        if i >= j:
          continue
        if self.edges.loc[i, j] == 0:
          continue
        plt.plot([self.X[i], self.X[j]], [self.Y[i], self.Y[j]],
                 color='red', alpha=self.edges.loc[i, j]/mx_weight)
    pads = self.if_pad[self.if_pad == 1].index
    gates = self.if_pad[self.if_pad == 0].index
    plt.scatter(self.X[pads].values, self.Y[pads].values, marker='s',
                zorder=10, color='black')
    plt.scatter(self.X[gates].values, self.Y[gates].values, marker='x',
                zorder=10)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.show()

  def dump_results(self, output_file):
    gates = self.if_pad[self.if_pad == 0].index
    results = pd.concat([self.X, self.Y], axis=1).loc[gates]
    results = results.sort_index()
    results = results.applymap(lambda x: '{:.8f}'.format(x))
    results.to_csv(output_file, sep=' ', header=None)


class QP:
  def __init__(self, input_file, output_file):
    self.read_input_file(input_file)
    self.set_edges()

    self.create_graph()
    self.graph.qp()
    self.graph.partition()
    self.graph.dump_results(output_file)

  def read_input_file(self, input_file):
    with open(input_file, 'r') as f:
      lines = f.readlines()

    n_gates, n_nets = lines[0].strip().split()
    n_gates, n_nets = int(n_gates), int(n_nets)

    X, Y = {}, {}
    net_to_gates = defaultdict(list)
    for i in range(1, n_gates + 1):
      items = list(map(int, lines[i].strip().split()))
      gate_id = items[0]
      net_ids = items[2:]
      for net_id in net_ids:
        net_to_gates[net_id].append(gate_id)
      X[gate_id] = random.randint(1, 100)
      Y[gate_id] = random.randint(1, 100)

    n_pads = int(lines[n_gates + 1].strip())
    pads = []
    for i in range(n_gates + 2, n_gates + n_pads + 2):
      items = list(map(int, lines[i].strip().split()))
      pad_id = items[0] + n_gates
      net_id = items[1]
      net_to_gates[net_id].append(pad_id)
      X[pad_id] = items[2]
      Y[pad_id] = items[3]
      pads.append(pad_id)

    self.net_to_gates = net_to_gates
    self.X = pd.Series(X).sort_index()
    self.Y = pd.Series(Y).sort_index()
    self.pads = pads
    self.n_gates = n_gates + n_pads

  def set_edges(self):
    edges = defaultdict(lambda: defaultdict(int))
    for net_id in self.net_to_gates:
      gate_ids = self.net_to_gates[net_id]
      for gate_id in gate_ids:
        for gate_id2 in gate_ids:
          if gate_id == gate_id2:
            continue
          edges[gate_id][gate_id2] += 1/(len(gate_ids) - 1)
          edges[gate_id2][gate_id] += 1/(len(gate_ids) - 1)
    edges = pd.DataFrame(edges).sort_index().T.sort_index().T.fillna(0)
    self.edges = edges

  def create_graph(self):
    self.graph = Graph(self.edges, self.pads, self.X, self.Y)

  def bft(self):
    visited = {k: 0 for k in self.edges.index}
    self.graph.bft(1, visited)
    return self.edges.shape[0] - pd.Series(visited).sort_index().sum()

  def dft(self):
    visited = {k: 0 for k in self.edges.index}
    self.graph.dft(1, visited)
    return self.edges.shape[0] - pd.Series(visited).sort_index().sum()

  def plot(self):
    self.graph.plot()

  def print_details(self):
    print('No. of pads:', len(self.pads))
    print('No. of gates:', self.n_gates - len(self.pads))
    print('Gate/Pad Coordinates:')
    print(pd.concat([self.X, self.Y], axis=1))


if __name__ == "__main__":
  with mp.Pool(processes=mp.cpu_count() - 2) as pool:
    args = [(f'./benchmarks/3QP/{f}', f'./benchmark solutions/3QP/{f}') for f in os.listdir('./benchmarks/3QP')]
    args.extend([(f'./benchmarks/8x8 QP/{f}', f'./benchmark solutions/8x8 QP/{f}') for f in os.listdir('./benchmarks/8x8 QP')])
    pool.starmap(QP, args)

