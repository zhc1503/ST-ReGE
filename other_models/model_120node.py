import torch
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool,SAGEConv
import torch.nn as nn
import torch.nn.functional as F

'''
The backbone for ST-ReGE as the number of node is 300.
The cases with 12, 120, 600, 1200 nodes are similar designed.
'''


class GCN300(torch.nn.Module):
    def __init__(self, dataset):
        super(GCN300, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 128)
        self.conv_ = GCNConv(128, 128)
        self.conv__ = GCNConv(128, 128)
        self.conv2 = GCNConv(128, 64)
        self.conv2_ = GCNConv(64, 64)
        self.conv2__ = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 32)
        self.conv3_ = GCNConv(32, 32)
        self.conv3__ = GCNConv(32, 32)
        self.conv4 = GCNConv(32, 16)
        self.conv4_ = GCNConv(16, 16)
        self.conv4__ = GCNConv(16, 16)
        self.conv5 = GCNConv(16, 8)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(120*8, 4)  # 300*4

        self.ffn1 = nn.Sequential(
            nn.Linear(250, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1000, 250),
            nn.BatchNorm1d(25),
            nn.Dropout(p=0.5)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.float()

        x = self.ffn1(x)  # FFN
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # # As the value of n is euqal to 1
        # s = x
        # x = self.conv_(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x += s

        # # As the value of n is euqal to 2
        # s = x
        # x = self.conv__(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x += s

        # # As the value of n is euqal to 3
        # s = x
        # x = self.conv__(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x += s

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        # # As the value of n is euqal to 1
        # s = x
        # x = self.conv2_(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x += s

        # # As the value of n is euqal to 2
        # s = x
        # x = self.conv2__(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x += s

        # # As the value of n is euqal to 3
        # s = x
        # x = self.conv2__(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x += s


        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        # # As the value of n is euqal to 1
        # s = x
        # x = self.conv3_(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x += s

        # # As the value of n is euqal to 2
        # s = x
        # x = self.conv3__(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x += s

        # # As the value of n is euqal to 3
        # s = x
        # x = self.conv3__(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x += s

        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

# # As the value of n is euqal to 1
        # s = x
        # x = self.conv4_(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x += s

        # # As the value of n is euqal to 2
        # s = x
        # x = self.conv4__(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x += s

        # # As the value of n is euqal to 3
        # s = x
        # x = self.conv4__(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x += s

        x = self.conv5(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        # x = self.avgpool(x)
        x = torch.reshape(x, (-1, 300*4))
        # fea = x  # t-SNE
        # x = torch.reshape(x, (-1, 300, 4))
        # x = torch.reshape(x, (-1, 12, 256))
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)

        x = self.fc(x)
        # return x, fea
        return x



