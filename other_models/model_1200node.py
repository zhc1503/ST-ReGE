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
        self.conv1 = GCNConv(dataset.num_features, 25)
        self.conv_ = GCNConv(25, 25)
        self.conv__ = GCNConv(25, 25)
        self.conv2 = GCNConv(25, 16)
        self.conv2_ = GCNConv(16, 16)
        self.conv2__ = GCNConv(16, 16)
        self.conv3 = GCNConv(16, 16)
        self.conv3_ = GCNConv(16, 16)
        self.conv3__ = GCNConv(16, 16)
        self.conv4 = GCNConv(16, 8)
        self.conv4_ = GCNConv(8, 8)
        self.conv4__ = GCNConv(8, 8)
        self.conv5 = GCNConv(8, 4)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(1200*4, 4)  # 300*4

        self.ffn1 = nn.Sequential(
            nn.Linear(25, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(100, 25),
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



