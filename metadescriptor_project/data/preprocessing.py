import os
from torch_geometric import datasets
import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from src.utils.graph_utils import safe_argmax
from src.utils.colors import RED, COLOR_OFF
from dotenv import load_dotenv

load_dotenv()
def get_valid_datasets(data_dir, blacklist=None):
    data_dir = os.getenv("root","data/TUDataset")
    if blacklist is None:
        blacklist = []

    dataset_names = sorted(os.listdir(data_dir))
    for bl in blacklist:
        if bl in dataset_names:
            dataset_names.remove(bl)

    valid_datasets = []
    print(dataset_names)
    for dataset in dataset_names:
        print(dataset)
        try:
            data = datasets.TUDataset(root=data_dir, name=dataset)
            if  data.num_edge_features==0:
                data = datasets.TUDataset(root=data_dir, name=dataset, transform=Add_One_Edge_Feature())
            if data.num_node_features * data.num_edge_features * len(data) == 0:
                print(f"{RED}[Erreur dataset vide] {dataset} n'a pas de features.{COLOR_OFF}")
                continue
            graph = to_networkx(data[0], to_undirected=True)

            atom_labels = safe_argmax(data[0].x, dim=1)
            if atom_labels is not None:
                for node in graph.nodes():
                    graph.nodes[node]['label'] = atom_labels[node].item()

            edge_labels = safe_argmax(data[0].edge_attr, dim=1)
            edge_index = data[0].edge_index.t().tolist()
            if edge_labels is not None:
                for (src, dst), label in zip(edge_index, edge_labels):
                    if graph.has_edge(src, dst):
                        graph[src][dst]['label'] = label.item()
                    else:
                        graph.add_edge(src, dst, label=label.item())

            valid_datasets.append(dataset)
        except Exception as e:
            print(f"[Erreur dataset complet] {e}")
            continue
    return valid_datasets
    
class Add_One_Edge_Feature(BaseTransform):
    def forward(self, data):
        return self.__call__(data)
    def __call__(self, data):
        num_edges = data.edge_stores[0]['edge_index'].size(1)
        data.edge_stores[0]['edge_attr'] = torch.ones((num_edges, 1), dtype=torch.float)
        return data

class GlobalStandardScaler(BaseTransform):
    def __init__(self, mean=None, std=None, fit=False):
        self.mean = mean
        self.std = std
        self.fit = fit  # Si True, calcule mean/std, sinon utilise ceux fournis

    def __call__(self, data):
        if self.fit:
            raise RuntimeError("Ne pas utiliser fit=True sur le test !")
        if self.mean is None or self.std is None:
            raise RuntimeError("Il faut d'abord fit sur le train.")
        data.y = (data.y - self.mean) / self.std
        return data

    @classmethod
    def fit_transform(cls, train_dataset):
        # Calcule mean/std sur le train
        all_y = torch.cat([data.y for data in train_dataset], dim=0)
        mean = all_y.mean(0, keepdim=True)
        std = all_y.std(0, unbiased=False, keepdim=True)
        print(std, mean)
        return cls(mean=mean, std=std, fit=False)
