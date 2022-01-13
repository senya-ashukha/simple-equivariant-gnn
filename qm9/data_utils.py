import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from qm9 import dataset
from qm9 import utils as qm9_utils

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from qm9 import dataset
from qm9 import utils as qm9_utils

class BatchGraph():
    def __init__(self, data, cuda, charge_scale):
        def generate_fc_edges():
            edges = []
            for graph_id in set(map(int, self.batch)):
                node_idx = np.arange(len(self.h))
                agraph = node_idx[self.batch==graph_id]
                comb_array = np.array(np.meshgrid(agraph, agraph)).T.reshape(-1, 2)
                comb_array = comb_array[comb_array[:,0] != comb_array[:,1]]
                edges.append(comb_array)  
            self.edges = torch.LongTensor(np.vstack(edges))
              
        self.batch = []
        self.y = data['homo']

        batch_size, n_nodes, _ = data['positions'].size()
        one_hot = data['one_hot']
        charges = data['charges']
        nodes = qm9_utils.preprocess_input(one_hot, charges, 2, charge_scale, 'cpu')
        self.h = nodes[data['atom_mask']]
        self.x = data['positions'][data['atom_mask']]

        batch = []
        for i, n in enumerate(data['num_atoms']):
            batch.append(np.ones(n)*i)
        self.batch = np.hstack(batch).astype(int)
        self.nG = batch_size

        generate_fc_edges()
        self.edges = self.edges
        self.batch = torch.LongTensor(self.batch)
        
        if cuda: self.cuda()
    
    def cuda(self):
        for k,v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                self.__dict__[k]=v.cuda()
        return self

    def __repr__(self):
        return f"""In the batch: num_graphs {self.nG} num_nodes {len(self.h)}
> .h \t\t a tensor of nodes representations \t\tshape {' x '.join(map(str, self.h.shape))}
> .x \t\t a tensor of nodes positions  \t\t\tshape {' x '.join(map(str, self.x.shape))}
> .edges \t a tensor of edges, a fully connected graph \tshape {' x '.join(map(str, self.edges.shape))}
> .batch  \t a tensor of graph_ids for each node \t\t{repr(self.batch)}
"""

def get_data(batch_size=96, num_workers=3):
    dataloaders, charge_scale = dataset.retrieve_dataloaders(batch_size, num_workers)
    return dataloaders['train'], dataloaders['valid'], dataloaders['test'], charge_scale