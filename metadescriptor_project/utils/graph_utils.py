from src.utils.colors import RED, COLOR_OFF
import torch
import networkx as nx
from torch_geometric.utils import to_networkx
from networkx.algorithms import isomorphism as iso
import matplotlib.pyplot as plt
import pdb

def safe_argmax(x, dim):
    if x is not None and x.dim() > dim and x.size(dim) > 0:
        return torch.argmax(x, dim=dim)
    return None

def open_graph(data):
    graph = to_networkx(data)

    if isinstance(graph, nx.DiGraph):
        graph = graph.to_undirected()
    atom_labels = safe_argmax(data.x, dim=1)
    if atom_labels is not None:
        for node in graph.nodes():
            graph.nodes[node]['label'] = atom_labels[node].item()
    else:
        print(f"{RED}Aucun label de nœud trouvé dans le graphe.{COLOR_OFF}")
        return (None, None)

    edge_labels = safe_argmax(data.edge_attr, dim=1)
    edge_index = data.edge_index.t().tolist()
    if edge_labels is not None:
        for (src, dst), label in zip(edge_index, edge_labels):
            if graph.has_edge(src, dst):
                graph[src][dst]['label'] = label.item()
            else:
                graph.add_edge(src, dst, label=label.item())
    else:
        print(f"{RED}Aucun label d'arête trouvé dans le graphe.{COLOR_OFF}")
        return (None, None)
    return (graph, data.y.item())

def is_subgraph_of(small, big, node_attr="label", edge_attr="label"):
    gm = iso.GraphMatcher(
        big,
        small,
        node_match=iso.categorical_node_match(node_attr, None),
        edge_match=iso.categorical_edge_match(edge_attr, None),
    )
    if hasattr(gm, "subgraph_is_monomorphic"):
        return gm.subgraph_is_monomorphic()
    else:
        return any(True for _ in gm.subgraph_monomorphisms_iter())

def closed_frequent_graphs_canonical(patterns, node_attr="label", edge_attr="label"):
    support_groups = {}
    for idx, (_, sup) in enumerate(patterns):
        support_groups.setdefault(sup, []).append(idx)

    return [
        (g_i, sup_i)
        for i, (g_i, sup_i) in enumerate(patterns)
        if not any(
            i != j and is_subgraph_of(g_i, patterns[j][0], node_attr, edge_attr)
            for j in support_groups[sup_i]
        )
    ]

def visualize_node_importance(model, data, device, title=""):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        _ = model(data.x, data.edge_index, data.batch)
        node_activations = model.last_node_embeddings
        importance = torch.norm(node_activations, dim=1).cpu().numpy()

    G = to_networkx(data, to_undirected=True)
    node_color = importance
    pos = nx.spring_layout(G)

    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    nodes = nx.draw_networkx_nodes(
        G, pos, node_color=node_color, cmap='coolwarm',
        node_size=300, ax=ax
    )
    nx.draw_networkx_edges(G, pos, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax)

    cbar = plt.colorbar(nodes, ax=ax)
    cbar.set_label("Node Importance (activation norm)")
    plt.title("Node Importance Visualization for " + title)
    plt.axis('off')
    plt.close()