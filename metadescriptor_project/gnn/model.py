import torch
from torch.nn import Linear, Dropout, Sequential
from torch.nn import BatchNorm1d as BatchNorm, ReLU
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, SAGEConv, GINConv, global_add_pool
import torch.nn.functional as F
import pdb
import numpy as np


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,num_layers=3,  dropout_rate=0.2, normalize=True):
        super(GNN, self).__init__()
        self.conv_layers = torch.nn.ModuleList([GCNConv(-1, hidden_channels, normalize=normalize)
                                             for _ in range(num_layers)])

        self.lin = Linear(hidden_channels, out_channels)
        self.dropout = Dropout(dropout_rate)
        self.last_node_embeddings = None

    def forward(self, x, edge_index, batch):
        for conv in self.conv_layers:
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)
        self.last_node_embeddings = x
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x
 

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, normalize=False):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = Sequential(
                Linear(in_channels, 2 * hidden_channels),
                BatchNorm(2 * hidden_channels),
                ReLU(),
                Linear(2 * hidden_channels, hidden_channels),
            )
            conv = GINConv(mlp, train_eps=True)

            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_channels))

            in_channels = hidden_channels

        self.lin1 = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.lin1(x)
        return x



class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.2, heads=8, normalize=False):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.conv_layers = torch.nn.ModuleList()
        self.conv_layers.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.conv_layers.append(GATConv(hidden_channels * heads, hidden_channels, 
                                          heads=heads, concat=True, dropout=dropout))
        

        self.conv_layers.append(GATConv(hidden_channels * heads, hidden_channels, 
                                      heads=heads, concat=True, dropout=dropout))
        
        self.linear = torch.nn.Linear(hidden_channels * heads, out_channels)
    def forward(self, x, edge_index, batch):
        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = global_mean_pool(x, batch)
        x = self.linear(x)
        return x





# def train(model, train_loader, optimizer, criterion, device):
#     model.train()
#     total_loss = 0
#     for data in train_loader:
#         data = data.to(device)
#         optimizer.zero_grad()
#         out = model(data.x, data.edge_index, data.batch)
#         #breakpoint()
#         loss = criterion(out, data.y)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * data.num_graphs
#     return total_loss / len(train_loader.dataset)



# def test(loader, model, device):
#     model.eval()
#     y_true = []
#     y_pred = []
#     y_probs = []
#     correct = 0
#     for data in loader:
#         data = data.to(device)
#         out = model(data.x, data.edge_index, data.batch)
#         probs = torch.softmax(out, dim=1)
#         pred = out.argmax(dim=1)
    
#         y_true.extend(data.y.cpu().numpy())
#         y_pred.extend(pred.cpu().numpy())
#         y_probs.extend(probs.cpu().detach().numpy() )
#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)
#     y_probs = np.array(y_probs)

#     # Calcul AUC pour chaque classe en mode multi-classes (one-vs-rest)
#     try:
#         print("attention l'auc est calculé pour du binaire")
#         #breakpoint()
#         auc = roc_auc_score(y_true, y_probs[:,1] )
#         #auc = roc_auc_score(y_true, y_probs)
#     except ValueError:
#         auc = None  # Pas calculable si pas assez de classes ou problème
    
#     results = {
#         'accuracy': accuracy_score(y_true, y_pred),
#         'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
#         'f1_score': f1_score(y_true, y_pred, average='weighted'),
#         'confusion_matrix': confusion_matrix(y_true, y_pred),
#         'classification_report': classification_report(y_true, y_pred, output_dict=True, zero_division=0),
#         'auc': auc
#     }

#     return results


# class GraphSAGE(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels,num_layers=3, drop_out=0.2):
#         super().__init__()
#         self.conv_layers = torch.nn.ModuleList([SAGEConv(-1, hidden_channels)
#                                              for _ in range(num_layers)])

#         self.lin = torch.nn.Linear(hidden_channels, out_channels)
#         self.drop_out = drop_out
#     def forward(self, x, edge_index, batch):
#         for conv in self.conv_layers:
#             x = F.relu(conv(x, edge_index))
#             x = self.dropout(x)
#         x = global_mean_pool(x, batch)  # Aggregate node features to graph features
#         x = self.lin(x)
#         return x

# def k_fold_reg(dataset, folds):
#     kf = KFold(n_splits=folds, shuffle=True, random_state=42)
#     train_indices, test_indices = [], []
#     val_indices = []                  
#     splits = list(kf.split(range(len(dataset))))                          
#     test_indices = [torch.tensor(test_idx, dtype=torch.long) for _, test_idx in splits]
#     val_indices = [test_indices[i - 1] for i in range(folds)]
#     for i in range(folds):
#         train_mask = torch.ones(len(dataset), dtype=torch.bool)
#         train_mask[test_indices[i]] = 0
#         train_mask[val_indices[i]] = 0
#         train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))
                                                                                       
#     return train_indices, test_indices, val_indices


# def k_fold(dataset, folds):
#     skf = StratifiedKFold(folds, shuffle=True, random_state=42)

#     test_indices, train_indices = [], []
#     for _, idx in skf.split(torch.zeros(len(dataset)), dataset.y):
#         test_indices.append(torch.from_numpy(idx).to(torch.long))

#     val_indices = [test_indices[i - 1] for i in range(folds)]

#     for i in range(folds):
#         train_mask = torch.ones(len(dataset), dtype=torch.bool)
#         train_mask[test_indices[i]] = 0
#         train_mask[val_indices[i]] = 0
#         train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

#     return train_indices, test_indices, val_indices
