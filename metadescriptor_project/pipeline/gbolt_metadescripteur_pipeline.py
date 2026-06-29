import sys
from pathlib import Path

# Add the root of your project to sys.path
project_root = Path(__file__).resolve().parent.parent.parent # Assuming your script is in src/pipeline
sys.path.insert(0, str(project_root))
import os
print(os.getcwd())
import pickle
import json
from torch_geometric import datasets
from src.pipeline.meta_descripteurs import MetaDescripteur_Dataset_GNX
from spmf import Spmf
from src.utils.io_utils import write_graphs_to_spmf, check_spmf_graph_file, spmf_to_networkX_with_freq, nb_pattern
from src.utils.graph_utils import open_graph, closed_frequent_graphs_canonical
from src.utils.colors import RED, YELLOW, COLOR_OFF
from src.data.preprocessing import get_valid_datasets, Add_One_Edge_Feature
from src.utils.my_columns_utils import rename_and_add_columns
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime
import re
import threading
import sys
import time

class TimeoutException(Exception):
    pass

def timeout_handler():
    raise TimeoutException("Temps d'exécution dépassé (45 minutes)")


timeout = 45 * 60  # 45 minutes en secondes



rename_dict_path = os.getenv('metadata')
columns_to_add_path = os.getenv('atome')
with open(rename_dict_path, "r") as f_json:
    rename_dict = json.load(f_json)

with open(columns_to_add_path, "rb") as f_pickle:
    columns_to_add = pickle.load(f_pickle)

def run_pipeline(DATASETS_NAMES):
    MIN_SUPS = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    MAX_NUM_EDGES = 99
    MAX_MEMORY = 150000
    print(DATASETS_NAMES)
    BLACK_LIST = list()
    dataset_dir = os.getenv("root","data/TUDataset")
    print(dataset_dir)
    if DATASETS_NAMES is None:
        print("NO DATASETS")
        sorted(os.listdir(dataset_dir))
    for dataset in DATASETS_NAMES:
        try:
            rename_dict[dataset]
        except:
            continue
        print(f"\n\n==== Processing dataset: {dataset} ====\n\n")
        print(dataset)
        ALL_DATA = {}
        DATA = datasets.TUDataset(root=dataset_dir, name=dataset, force_reload=False)
        if  DATA.num_edge_features==0:
                DATA = datasets.TUDataset(root=dataset_dir, name=dataset, transform=Add_One_Edge_Feature())
        DATASET = [open_graph(DATA[i]) for i in range(len(DATA))]
        start_len = len(DATASET)
        DATASET = [g for g in DATASET if g[0] is not None and g[0].number_of_nodes() > 1 and g[0].number_of_edges() > 0]
        if len(DATASET) == 0:
            print(f"{RED}[Erreur dataset vide] Le dataset {dataset} n'a pas de graphes valides.{COLOR_OFF}")
            continue
        if len(DATASET) < start_len:
            print(f"{YELLOW}[Attention] {start_len - len(DATASET)} graphes supprimés.{COLOR_OFF}")

        infos = MetaDescripteur_Dataset_GNX.Simple.compute_graph_descriptor_all(DATASET)
        infos = rename_and_add_columns(infos , rename_dict[dataset],  columns_to_add)
        stats = MetaDescripteur_Dataset_GNX.Statistics.compute_data_statistics(infos)
        ALL_DATA[f"SPMF/{dataset}_NORMAL_infos"] = infos
        ALL_DATA[f"SPMF/{dataset}_NORMAL_stats"] = stats

        write_graphs_to_spmf([g[0] for g in DATASET], f"SPMF/{dataset}.spmf", sep_graph="")
        #breakpoint()
        check_spmf_graph_file(f"SPMF/{dataset}.spmf")

        for min_sup in MIN_SUPS:
            print("GSPAN")
            print(min_sup)
            location = "SPMF"
            FILE_GSPAN_NAME = f"SPMF/{dataset}_GSPAN_{min_sup}.txt"
            timer = threading.Timer(timeout, lambda: timeout_handler())
            timer.start()
            try :
                if not os.path.exists(FILE_GSPAN_NAME) or os.path.getsize(FILE_GSPAN_NAME) == 0:
                    #os.system(f"./../../gBolt/build/gbolt -i SPMF/{dataset}.spmf -s {min_sup} -o {FILE_GSPAN_NAME} -d")
                    os.system(f"./../../SEMA-quickspan/build/gspan -input_file SPMF/{dataset}.spmf -support {min_sup}  -output_file {FILE_GSPAN_NAME} -pattern true")
                    gbolt_files = [
              f for f in os.listdir("SPMF/")
              if re.search(fr'{dataset}_GSPAN_{min_sup}.txt.t\d*$', f, re.I)
                    ]         
                    print("gbolt",gbolt_files)
                    concat_files(gbolt_files, FILE_GSPAN_NAME)
                    remove_files(gbolt_files)
            except TimeoutException as e:
                print(f"ERREUR: {e}", file=sys.stderr)
                continue
            finally:
                # Arrêter le timer si le code se termine avant
                timer.cancel()

            DATASET_CGSPAN_MY = spmf_to_networkX_with_freq(FILE_GSPAN_NAME)
            CLOSED_CGSPAN_MY = closed_frequent_graphs_canonical(DATASET_CGSPAN_MY)
            FILE_CGSPAN_NAME = f"SPMF/{dataset}_CGSPAN_{min_sup}.txt"
            write_graphs_to_spmf(CLOSED_CGSPAN_MY, FILE_CGSPAN_NAME, with_freq=True, sep_graph="")
            check_spmf_graph_file(FILE_CGSPAN_NAME)

            infos_GSPAN = MetaDescripteur_Dataset_GNX.Simple.compute_graph_descriptor_all(DATASET_CGSPAN_MY)
            infos_GSPAN = rename_and_add_columns(infos_GSPAN , rename_dict[dataset],  columns_to_add)
            stats_GSPAN = MetaDescripteur_Dataset_GNX.Statistics.compute_data_statistics(infos_GSPAN)
            ALL_DATA[f"SPMF/{dataset}_GSPAN_{min_sup}_infos"] = infos_GSPAN
            ALL_DATA[f"SPMF/{dataset}_GSPAN_{min_sup}_stats"] = stats_GSPAN

            infos_CGSPAN = MetaDescripteur_Dataset_GNX.Simple.compute_graph_descriptor_all(CLOSED_CGSPAN_MY)
            infos_CGSPAN = rename_and_add_columns(infos_CGSPAN , rename_dict[dataset],  columns_to_add)
            stats_CGSPAN = MetaDescripteur_Dataset_GNX.Statistics.compute_data_statistics(infos_CGSPAN)
            ALL_DATA[f"SPMF/{dataset}_CGSPAN_{min_sup}_infos"] = infos_CGSPAN
            ALL_DATA[f"SPMF/{dataset}_CGSPAN_{min_sup}_stats"] = stats_CGSPAN

        with open(f"SPMF/{dataset}_infos.pkl", "wb") as f:
            pickle.dump(ALL_DATA, f)

def remove_files(files_list, location="SPMF/"):
    for name in files_list:
        try:
            path = os.path.join(location, name)
            os.remove(path)
        except FileNotFoundError:
            print(f"{path} not found.")

def concat_files(files_list, new_file_name, location="SPMF/"):
    with open(new_file_name, "w") as new_file:
        for name in files_list:
            path = os.path.join(location, name)
            with open(path) as f:
                for line in f:
                    new_file.write(line)


if __name__ == "__main__":
    DATASETS_NAMES = {
     'BZR',
    'BZR_MD',
    'COX2',
    'COX2_MD',
    'DHFR',
     'DHFR_MD',
    'ER_MD',
    'NCI-H23H',
    'NCI1',
    'NCI109',
    #'ZINC_full',
    'ZINC_test',
    #'ZINC_train',
    'ZINC_val',
    'aspirin',
    #'benzene',
    'ethanol',
    'malonaldehyde',
    'naphthalene',
    'salicylic_acid',
    'toluene',
    'uracil'}

    run_pipeline(DATASETS_NAMES)
