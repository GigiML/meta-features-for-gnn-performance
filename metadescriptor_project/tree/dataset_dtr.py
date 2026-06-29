from itertools import product
import os
import polars as pl


class Dataset:
    @staticmethod
    def get_all_features():
        stat = ["min", "mean", "std", "skewness", "kurtosis", "max"]

        f1 = ["num_nodes", "num_edges", "num_etiquette_sommets_differents", "diameter", "density"]
        f2 = [
            "wiener_index", "clustering_coef", "traingle", "kcore_max",
            "mean_degree_centrality", "mean_betweenness", "mean_pagerank", "mean_closeness"
        ]
        f3 = ["fiedler_value", "spectral_radius", "trace_laplacien", "nb_zero_eigenvalues"]
        other = ["Count"]
        class_f = ["dataset_nb_classes", "dataset_prop_classe_majorite", "dataset_label_entropy"]

        thresh = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

        curr_f1 = f1 
        curr_f2 = f2 
        curr_f3 = f3 

        features = curr_f1 + curr_f2 + curr_f3

        class_feature = other  + class_f 
        itertool_normal = [
            f"NORMAL_{feat}_{s}"
            for feat, s in product(features, stat)
        ]

        itertool_gspan = [
            f"{algo}_{thre}_{feat}_{s}"
            for (algo, thre), feat, s in product(
                product(["GSPAN", "CGSPAN"], thresh),
                features + ["frequence"],
                stat
            )
        ]

        itertool2 = [
            f"NORMAL_{feat}"
            for feat in class_feature
        ] + [
            f"{algo}_{thre}_{feat}"
            for (algo, thre), feat in product(
                product(["GSPAN", "CGSPAN"], thresh),
                class_feature
            )
        ]
        return itertool_normal + itertool_gspan + itertool2
    
    @staticmethod
    def get_feature_ablation(family, source):
        """Renvoie TOUTES les features SAUF la combinaison donnée"""
        all_features = Dataset.get_all_features()  
        one_combination = Dataset.get_one_combination(family, source) 
        ablation_features = set(all_features) - set(one_combination)
        return sorted(ablation_features) 



    @staticmethod
    def get_one_combination(
        family="F1",
        source="NORMAL", 
    
    ):
        stat = ["min", "mean", "std", "skewness", "kurtosis", "max"]

        f1 = ["num_nodes", "num_edges", "num_etiquette_sommets_differents", "diameter", "density"]
        f2 = [
            "wiener_index", "clustering_coef", "traingle", "kcore_max",
            "mean_degree_centrality", "mean_betweenness", "mean_pagerank", "mean_closeness"
        ]
        f3 = ["fiedler_value", "spectral_radius", "trace_laplacien", "nb_zero_eigenvalues"]
        f4 = ["dataset_nb_classes", "dataset_prop_classe_majorite", "dataset_label_entropy"]
        other = ["Count"]
        thresh = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

        families = {
            "F1": f1,  
            "F2": f2,
            "F3": f3,
            "F4": f4,
        }
        if family is None:
            return []
        if family not in families:
            raise ValueError(f"Famille inconnue: {family}")

        feats = families[family]

        if source == "NORMAL":
            if family == "F4":
                return [f"NORMAL_{feat}" for feat in feats]
            elif family == "F1":
                return  [f"NORMAL_{feat}_{s}" for feat, s in product(feats, stat)] + [f"NORMAL_Count"]
            else:
                return [f"NORMAL_{feat}_{s}" for feat, s in product(feats, stat)]

        if source in {"GSPAN", "CGSPAN"}:
            if family == "F4":
                return [f"{s}_{t}_{feat}" for s, t, feat in product(["GSPAN", "CGSPAN"],thresh,feats)]
            elif family == "F1":
                return  [f"{s}_{t}_{feat}_{st}" for s, t, feat, st in product(["GSPAN", "CGSPAN"], thresh,feats, stat)] + [f"{s}_{t}_Count" for s, t in product(["GSPAN", "CGSPAN"],thresh)]
            else:
                extra_feats = feats #+ ["frequence"]
                return [f"{s}_{t}_{feat}_{st}" for s, t, feat, st in product(["GSPAN", "CGSPAN"],thresh, extra_feats, stat)]

    @staticmethod
    def get_performance(name_file, directory=".", perf="Accuracy"):
        file_path = os.path.join(directory, name_file)
        df = pl.read_csv(file_path)
        cols = ["Dataset", f"{perf}"]#, f"Worst_{perf}", f"Best_{perf}", f"Std_{perf}"]
        return df.select(cols)

    @staticmethod
    def get_dataset(
        summary_file="summary.csv",
        perf_file=None,
        directory=".",
        perf="Accuracy",
        family="F1",
        source="NORMAL",
        ablation=True, 
    ):
        summary_path = os.path.join(directory, summary_file)
        df = pl.read_csv(summary_path)
        print("ablation:", ablation)
        if ablation:
            selected_features  = Dataset.get_feature_ablation(family, source)
        else:
            selected_features = Dataset.get_one_combination(family, source)
      

        available_features = [c for c in selected_features if c in df.columns]
        df = df.select(["Dataset"] + available_features)

        if perf_file is None:
            name_dataset = df["Dataset"]
            columns = df.drop("Dataset").columns
            X = df.drop("Dataset").to_numpy()
            return name_dataset, columns, X

        df_perf = Dataset.get_performance(name_file=perf_file, directory=directory, perf=perf)

        X_df, y_df = pl.align_frames(df, df_perf, on="Dataset", how="inner")

        name_dataset = X_df["Dataset"]
        columns = X_df.drop("Dataset").columns
        columns_perf = y_df.drop("Dataset").columns

        X = X_df.drop("Dataset").to_numpy()
        y = y_df.drop("Dataset").to_numpy()

        return name_dataset, columns, columns_perf, X, y
    


# name_dataset, columns, columns_perf, X, y = Dataset.get_dataset(
# summary_file="summary.csv",
# perf_file="performance.csv",
# directory=".",
# perf="Accuracy",
# use_f1=True,
# use_f2=True,
# use_f3=False,
# use_class_f=True
# )