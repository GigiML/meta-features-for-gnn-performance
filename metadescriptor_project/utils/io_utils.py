from src.utils.colors import RED, COLOR_OFF

def print_beautiful_dict(dictionary):
    for key, value in dictionary.items():
        print(f"{key}: {value}")

def check_spmf_graph_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()##### très mauvais pour la ram faire un yield à la place
    ## check le format
    for line in lines:
        line = line.strip()
        if not line:
            continue

        tokens = line.split(" ")

        if tokens[0] == 't':
            if (len(tokens) != 3 and len(tokens) != 5) or tokens[1] != '#':
                print(f"{RED}Erreur dans la ligne de titre : '{line}'{COLOR_OFF}")
                return

        elif tokens[0] == 'v':
            if len(tokens) != 3:
                print(f"{RED}Erreur dans la ligne de nœud : '{line}'{COLOR_OFF}")
                return

        elif tokens[0] == 'e':
            if len(tokens) != 4:
                print(f"{RED}Erreur dans la ligne d'arête : '{line}'{COLOR_OFF}")
                return
    #check le fond
    with open(filename, "r") as f:
        lines = f.readlines()

    current_nodes = set()
    graph_id = None
    for line in lines:
        tokens = line.strip().split()
        if not tokens:
            continue
        if tokens[0] == "t":
            current_nodes = set()
            graph_id = tokens[-1]
        elif tokens[0] == "v":
            node_id = int(tokens[1])
            current_nodes.add(node_id)
        elif tokens[0] == "e":
            src = int(tokens[1])
            tgt = int(tokens[2])
            if src not in current_nodes or tgt not in current_nodes:
                print(f"Graphe {graph_id} : arête invalide e {src} {tgt} - sommet manquant")
                return

def write_graphs_to_spmf(graphs, filename, with_freq=False, sep_graph="\n"):
    import networkx as nx
    with open(filename, 'w') as f:
        for i, item in enumerate(graphs):
            if with_freq:
                g, freq = item
            else:
                g = item
                freq = None

            if g.number_of_nodes() == 0:
                continue

            if with_freq:
                f.write(f"t # {i} * {freq}\n")
            else:
                f.write(f"t # {i}\n")

            for nid, attr in g.nodes(data=True):
                label = attr.get('label', 0)
                f.write(f"v {nid} {int(label) + 1}\n")

            for src, dst, attr in g.edges(data=True):
                label = attr.get('label', 0)
                f.write(f"e {int(src)} {int(dst)} {int(label) + 1}\n")

            f.write(sep_graph)

def spmf_to_networkX_with_freq(filename):
    import networkx as nx
    graphs = []
    with open(filename, 'r') as f:
        lines = f.readlines()###LA RAM
        current_graph = None
        for line in lines:
            if line.startswith("t #"):
                if current_graph is not None:
                    graphs.append((current_graph, frequence))
                current_graph = nx.Graph()
                frequence = int(line.split(' ')[4])
            elif line.startswith("v"):
                parts = line.split()
                current_graph.add_node(int(parts[1]), label=parts[2])
            elif line.startswith("e"):
                parts = line.split()
                current_graph.add_edge(int(parts[1]), int(parts[2]), label=parts[3])
        if current_graph is not None:
            graphs.append((current_graph, frequence))
    return graphs

def nb_pattern(patterns):
    count = 0
    for pattern in patterns:
        if pattern[0].startswith("t #"):
            count += 1
    return count