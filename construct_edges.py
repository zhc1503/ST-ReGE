import torch

def construct_edges(inter_sys_connect, dataset):
    '''
    construct edges according to DNA of an individual
    :param inter_sys_connect: (inter_sys_connect_size, )
    :return: dataset

    example:
    inter_sys_connect: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1]
    lead I aVF V5 V6 are selected for connections between limb and chest lead systems
    '''
    num_graph = dataset.data.y.shape[0]
    edge_index = []
    edge_index_final = []
    # Firstly, connect the patches belonging to the same lead sequentially.
    for i in range(12):  # number of leads
        for j in range(25):  # number of patches per lead
            # The first and the last patch only has 1 neighbor
            if j == 0:
                edge_index.append([i * 25 + j, i * 25 + j + 1])
            elif j == 24:
                edge_index.append([i * 25 + j, i * 25 + j - 1])
            else:
                edge_index.append([i * 25 + j, i * 25 + j + 1])
                edge_index.append([i * 25 + j, i * 25 + j - 1])
    # Secondly, connect leads within the same lead system. (limb/chest)
    for j in range(25):  # number of patches per lead
        for i in range(6):  # number of leads within each lead system
            for k in range(5):
                edge_index.append([i * 25 + j, 25 * ((i + k + 1) % 6) + j])  # limb lead
            for k in range(5):
                edge_index.append([i * 25 + j, 25 * ((i + k + 1) % 6 + 6) + j])  # chest lead
        # Finally, connect the leads across the limb and chest system
        connected_ls = []
        for i in range(inter_sys_connect.shape[0]):
            if inter_sys_connect[i] == 1:
                connected_ls.append(i)
        for i in range(len(connected_ls)):
            for k in range(len(connected_ls)):
                if i != k:
                    edge_index.append([25*connected_ls[i]+j, 25*connected_ls[k]+j])
    num_edges_per_graph = len(edge_index)
    for num in range(num_graph):
        edge_index_final.append(edge_index)
    edge_index = torch.tensor(edge_index_final, dtype=torch.long)
    edge_index = torch.reshape(edge_index, (-1, 2))
    edge_index = edge_index.t().contiguous()
    dataset.data.edge_index = edge_index
    edge_index_slice = []
    for i in range(dataset.slices['y'].shape[0]):
        edge_index_slice.append(i * num_edges_per_graph)
    edge_index_slice = torch.tensor(edge_index_slice)
    dataset.slices['edge_index'] = edge_index_slice
    return dataset