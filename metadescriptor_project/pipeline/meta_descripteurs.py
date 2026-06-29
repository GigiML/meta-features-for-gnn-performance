import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import entropy
from collections import Counter


class MetaDescripteur_GNX:
    class Simple:

        @staticmethod
        def Nr_instances(graph):
            """Retourne le nombre de nœuds dans le graphe."""
            return graph.number_of_nodes()
        
        @staticmethod
        def Nr_edges(graph):
            """Retourne le nombre d'arêtes dans le graphe."""
            return graph.number_of_edges()
        
        @staticmethod
        def Nr_classes(graph, label):
            """Retourne le nombre de classes distinctes dans le graphe.
            Peux s'apparenter au nombre d'atome différents dans un graphe moléculaire."""
            return len(set(nx.get_node_attributes(graph, label).values()))
        
        @staticmethod
        def Nr_features(graph):
            """Retourne le nombre de caractéristiques (attributs) des nœuds."""
            return len(next(iter(graph.nodes(data = True)))[1])
        @staticmethod

        def count_nodes_per_label(G):
            labels = nx.get_node_attributes(G, 'label').values()  # Récupère les labels de tous les nœuds
            label_counts = Counter(labels)  # Compte combien de fois chaque label apparaît
            return dict(label_counts)


class MetaDescripteur_Dataset_GNX:
    class Simple:
        ######### Mesures simples #########

        @staticmethod
        def Nr_nodes(graphs):
            """Retourne le nombre de nœuds dans le graphe."""
            mean = 0
            for graph in graphs:
                mean += graph[0].number_of_nodes()
            return mean / len(graphs)
        
        @staticmethod
        def Nr_edges(graphs):
            """Retourne le nombre d'arêtes dans le graphe."""
            mean = 0
            for graph in graphs:
                mean += graph[0].number_of_edges()
            return mean / len(graphs)

        @staticmethod
        def Nr_classes(graphs, label):
            """Retourne le nombre de classes distinctes dans le graphe."""
            mean = 0
            for graph in graphs:
                mean += len(set(nx.get_node_attributes(graph[0], label).values()))
            return mean / len(graphs)
        
        @staticmethod
        def Nr_features(graphs):
            """Retourne le nombre de caractéristiques (attributs) des nœuds."""
            mean = 0
            for graph in graphs:
                mean += len(next(iter(graph[0].nodes(data = True)))[1])
            return mean / len(graphs)
        
        @staticmethod
        def missing_values(graphs, label):
            """Retourne le nombre de valeurs manquantes dans le graphe."""
            mean = 0
            for graph in graphs:
                mean += len([node for node in graph[0].nodes() if label not in graph[0].nodes[node]])
            return mean / len(graphs)
        
        @staticmethod
        def Nr_instances(graphs):
            """Retourne le nombre d'instances (lignes) dans le graphe."""
            return len(graphs)
        
        
        @staticmethod
        def compute_graph_descriptors(G):
            # ── global ────────────────────────────────────────
            num_nodes = G.number_of_nodes()
            num_edges = G.number_of_edges()
            # num_classes = len(set(nx.get_node_attributes(G, 'label').values()))
            density = nx.density(G)
            if nx.is_connected(G):
                diameter = nx.diameter(G)
                wiener_index = nx.wiener_index(G)
            else:
                diameter = np.nan
                wiener_index = np.nan
            triangles = sum(nx.triangles(G).values())

            clustering_coef = nx.average_clustering(G)
            kcore_max = np.max(list(nx.core_number(G).values()))

            # ── centralité ────────────────────────────────────────
            degrees = np.array([d for n, d in G.degree()])
            mean_degree_centrality = degrees.mean()
  


            betweenness = np.array(list(nx.betweenness_centrality(G).values()))
            mean_betweenness = betweenness.mean()

            pagerank = np.array(list(nx.pagerank(G).values()))
            mean_pagerank = pagerank.mean()

            closeness = np.array(list(nx.closeness_centrality(G).values()))
            mean_closeness = closeness.mean()

            # ── spectral ────────────────────────────────────────
            L = nx.normalized_laplacian_matrix(G).todense()
            eigvals = np.linalg.eigvalsh(L)
            eigvals = np.sort(np.real(eigvals))

            spectral_radius = eigvals[-1]
            fiedler_value = eigvals[1] if len(eigvals) > 1 else 0
            nb_zero_eigenvalues = np.sum(np.isclose(eigvals, 0, atol=1e-8))
            trace_laplacian = 2 * G.number_of_edges()

            #labels = nx.get_node_attributes(G, 'label').values() 
            #label_counts = Counter(labels)  # Compte combien de fois chaque label apparaît
            #label_counts = dict(label_counts)
            # return {
            #     'num_nodes': num_nodes,
            #     'num_edges': num_edges,
            #     'num_label_differents': num_classes,
            #     'density': density,
            #     'diameter': diameter,

            #     #**label_counts
            # }
            return {
                    # ── global ──
                'num_nodes': num_nodes,
                'num_edges': num_edges,
                'density': density,
                'diameter': diameter,
                'wiener_index': wiener_index,
                'clustering_coef': clustering_coef,
                'traingle': triangles,
                'kcore_max': kcore_max,

                # ── local ──
                'mean_degree_centrality': mean_degree_centrality,
                'mean_betweenness': mean_betweenness,
                'mean_pagerank': mean_pagerank,
                'mean_closeness': mean_closeness,

                # ── spectral ──
                'fiedler_value': fiedler_value,
                'spectral_radius': spectral_radius,
                'trace_laplacien': trace_laplacian,
                'nb_zero_eigenvalues': nb_zero_eigenvalues,

             }
        @staticmethod
        def compute_graph_descriptors_by_label(Gs, label):
            num_nodes = 0
            num_edges = 0
            max_distance = 0
            for G in Gs:
                try:
                    num_nodes += len([node for node in G.nodes() if G.nodes[node]['label'] == label])
                except KeyError:
                    print(f"Label '{label}' non trouvé dans les attributs des nœuds du graphe.")
                    continue
                # compte le nombre d'arrêtes qu'a chaque noeud
                liste_edges = [G.edges(node) for node in G.nodes() if G.nodes[node]['label'] == label] # list[list]
                num_edges += sum([len(edges) for edges in liste_edges])
                for node in G.nodes():
                    if G.nodes[node]['label'] != label:
                        continue
                    try:
                        distance = nx.single_source_shortest_path_length(G, node)
                        max_distance = max(max_distance, max(distance.values()))
                    except nx.NetworkXNoPath:
                        continue

            return {
                'num_nodes': num_nodes,
                'num_edges': num_edges,
                'max_distance': max_distance,
            }

        @staticmethod
        def compute_classification_dataset_stats(graphs):
            """Calcule les stats classification pour TOUT le dataset (pas graphe par graphe)."""
            
            #all_labels = []
            
            # for graph, _ in graphs:
            #     labels_attr = nx.get_node_attributes(graph, 'label')
            #     for node_labels in labels_attr.values():
            #         if isinstance(node_labels, (list, set, tuple)):
            #             all_labels.extend(node_labels)
            #         else:
            #             all_labels.append(node_labels)
            all_labels = [label for _, label in graphs]
            if not all_labels:
                return {'dataset_nb_classes': 0,
                        'dataset_prop_classe_majorite': 0,
                        'dataset_label_entropy': 0,
                         }
            # Stats GLOBALES sur tout le dataset
            label_counts = Counter(all_labels)
            total_labels = len(all_labels)
            dataset_nb_classes = len(label_counts.values())
            dataset_prop_classe_majorite = max(label_counts.values()) / total_labels
            dataset_label_entropy = entropy(list(label_counts.values()), base=2)
            stats = {'dataset_nb_classes': dataset_nb_classes,
                    'dataset_prop_classe_majorite': dataset_prop_classe_majorite,
                    'dataset_label_entropy': dataset_label_entropy,}
            return stats
        
        @staticmethod
        def compute_graph_descriptor_all(graphs):
            """Calcule les descripteurs pour une liste de graphes avec labels arbitraires (int ou str)."""
            stats_globales =  MetaDescripteur_Dataset_GNX.Simple.compute_classification_dataset_stats(graphs)
            descriptors = [
                {**MetaDescripteur_Dataset_GNX.Simple.compute_graph_descriptors(graph[0]), 'label': graph[1]}
                for graph in graphs
            ]
            df = pd.DataFrame.from_records(descriptors)
            stats_df = pd.DataFrame([stats_globales] * len(df)).reset_index(drop=True)
            df = df.reset_index(drop=True)
            return pd.concat([df, stats_df], axis=1) 
        @staticmethod
        def compute_graph_descriptor_all_for_all_label(graphs):
            liste_label_dict = {}
            for graph in graphs:
                label_graph_x = set(nx.get_node_attributes(graph[0], 'label').values())
                for label in label_graph_x:
                    if label not in liste_label_dict:
                        liste_label_dict[label] = 0
                    liste_label_dict[label] += 1
            liste_label = list(liste_label_dict.keys())
            descripteurs = [
                {**MetaDescripteur_Dataset_GNX.Simple.compute_graph_descriptors_by_label([g[0] for g in graphs], label), 'occurence': liste_label_dict[label] / len(graphs), 'label': label}
                for label in liste_label
            ]
            return pd.DataFrame.from_records(descripteurs)

    class Statistics:
        @staticmethod
        def compute_data_statistics(data):
            """
            Calcule les statistiques de base pour chaque colonne numérique du DataFrame.
            :param data: DataFrame
            :return: DataFrame avec les statistiques
            """
            stats = {
                'min': data.min(numeric_only=True),
                'kurtosis': data.kurtosis(numeric_only=True),
                'skewness': data.skew(numeric_only=True),
                'std': data.std(numeric_only=True),
                'mean': data.mean(numeric_only=True),
                'max': data.max(numeric_only=True),
            }
            return pd.DataFrame(stats).T.replace([np.inf, -np.inf, np.nan], 0)

