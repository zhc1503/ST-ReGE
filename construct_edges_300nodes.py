import torch
import os.path as osp
from torch_geometric.data import Data, Dataset, InMemoryDataset
import numpy as np
'''
An example to construct ST-ReGE with 300 nodes
'''

edge_index = []
# Firstly, connect the patches belonging to the same lead sequentially.
for i in range(8):
    for j in range(25):
        if j == 0:
            edge_index.append([i*25+j, i*25+j+1])
        elif j == 24:
            edge_index.append([i*25+j, i*25+j-1])
        else:
            edge_index.append([i*25+j, i*25+j+1])
            edge_index.append([i*25+j, i*25+j-1])

# Secondly, connect leads within the same lead system. (limb/chest)
for j in range(25):
    for i in range(2):
        edge_index.append([i*25+j, 25*((i+1) % 2)+j])
    for i in range(6):
        edge_index.append([(i+2) * 25 + j, 25 * ((i + 1) % 6 + 2) + j])
        edge_index.append([(i+2) * 25 + j, 25 * ((i + 2) % 6 + 2) + j])
        edge_index.append([(i+2) * 25 + j, 25 * ((i + 3) % 6 + 2) + j])
        edge_index.append([(i+2) * 25 + j, 25 * ((i + 4) % 6 + 2) + j])
        edge_index.append([(i+2) * 25 + j, 25 * ((i + 5) % 6 + 2) + j])
    # Finally, connect the leads across the limb and chest system
    edge_index.append([j, 25*5+j])
    edge_index.append([j, 25*6+j])
    edge_index.append([25*5+j, j])
    edge_index.append([25*6+j, j])
    edge_index.append([25*1+j, 25*5+j])
    edge_index.append([25*1+j, 25*6+j])
    edge_index.append([25*5+j, 25*1+j])
    edge_index.append([25*6+j, 25*1+j])
edge_index = torch.tensor(edge_index, dtype=torch.long)
edge_index = edge_index.t().contiguous()