import pickle
import polars as pl
from sklearn.model_selection import KFold, LeaveOneOut
from src.tree.dataset_dtr import Dataset
from src.analysis.feature_importance import FeatureAnalysisConfig, _train_models_comparison


class DecisionTreeAnalyzer:
    def __init__(
        self,
        model="GCN",
        summary_file="summary.csv",
        target_file="results_classif_logging1.csv",
        directory=".",
        perf="Accuracy",
        depth_max=10,
        cv_type="loo",
        n_splits=10,
        family="F1", 
        source="NORMAL", 
        ablation=True,
        config=None,
    ):
        self.model = model
        self.summary_file = summary_file
        self.target_file = target_file
        self.directory = directory
        self.perf = perf
        self.depth_max = depth_max
        self.cv_type = cv_type
        self.n_splits = n_splits
        self.family=family
        self.source=source
        self.ablation=ablation

        self.config = config if config is not None else FeatureAnalysisConfig()

    def train(self):
        name_dataset, columns, columns_perf, X, y = Dataset.get_dataset(
            summary_file=self.summary_file,
            perf_file=self.target_file,
            directory=self.directory,
            perf=self.perf,
            family=self.family, 
            source=self.source, 
            ablation=self.ablation
           
        )

        cv_results, maes = [],[]

        if self.cv_type.lower() == "loo":
            splitter = LeaveOneOut()
        elif self.cv_type.lower() == "kfold":
            splitter = KFold(n_splits=self.n_splits, shuffle=False)
        elif self.cv_type.lower() == "all":
            res, mae = _train_models_comparison(
                X, X,
                y, y,
                columns, columns_perf,
                self.config,
                None, "", None, True,
                cv=0,
                model_name=self.model,
                ablation=self.ablation,
                family=self.family,
                source=self.source
            )
            cv_results.append(res)
            maes.append(mae)
            return cv_results, mae,  name_dataset
        else:
            raise ValueError("cv_type doit être 'loo', 'kfold' ou 'all'")

        for i, (train_index, test_index) in enumerate(splitter.split(X)):
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]

            res, mae = _train_models_comparison(
                X_train, X_test,
                y_train, y_test,
                columns, columns_perf,
                self.config,
                None, "", None, True,
                cv=i,
                model_name=self.model,
                family=self.family,
                source=self.source,
                ablation=self.ablation,
            )
            print(i, mae, name_dataset[i])
            cv_results.append(res)
            maes.append(mae)

        return cv_results, maes, name_dataset

    def report_feature_importances(self, max_cv=1,summary_file_column=None):
        columns_perf = [
            f"{self.perf}"
        ]

        rows = []
        depth=10
        print(self.family, self.source)
        for cv in range(max_cv):
            filename = (
                f"models/multi_model_{self.model}_depth{depth}_"
                f"{self.perf}_{self.source}_{self.family}_{self.ablation}_{cv}.pkl"
            )

            with open(filename, "rb") as f:
                model = pickle.load(f)

            for i, column in enumerate(columns_perf):
                line = [column, depth, cv]
                line.extend(model.estimators_[i].feature_importances_)
                rows.append(line)

        df = pl.DataFrame(
            rows,
            schema=["column", "depth", "cv"] + list(summary_file_column),
            strict=False,
            orient="row"
        )

        grouped_df = (
            df.drop("cv")
            .group_by(["column", "depth"], maintain_order=True)
            .sum()
        )

        args_str = (
            f"{self.family}_{self.source}_{self.ablation}"
        )
        output_file = f"feature_importances_{self.model}_{self.perf}_{max_cv}_{args_str}.csv"

        
        grouped_df = (
            grouped_df
            .drop("column")
            .transpose(include_header=True, header_name="features")
            .with_columns(
                pl.col("column_0").cast(pl.Float64)
            )
        )
        grouped_df.sort("column_0").write_csv(output_file)
        

        df_count =df.with_columns(  pl.all().map_elements( lambda x: None if x == 0.0 else x)).count().transpose(include_header=True, header_name="features").sort("column_0")
        df_count.write_csv(f"Countfeature_importances_{self.model}_{self.perf}_{max_cv}_{args_str}.csv")


        return output_file
    
    def save_cv_results(self, rows, perf):
        """
        À partir du DataFrame `df` avec:
          - 'name_dataset': Series
          - 'cv_results': list de dict {depth: array(perfs), ...}
        construit un DataFrame avec:
          ['Dataset', 'Mean_{perf}', 'Worst_{perf}', 'Best_{perf}', 'Std_{perf}']
        et exporte en CSV.
        """
        cols = ["depth", f"{perf}"]

        data = [
            {
                "Depth": int(depth),
                perf : accs[0]
            }
            for depth, accs in rows[0].items()
        ]
        df_cv = pl.DataFrame(
            data,
            schema=cols,
            strict=False,
        )

        args_str = (
            f"{self.family}_{self.source}_{self.ablation}"
        )

        output_file = f"result_DT_{self.model}_{self.perf}_{args_str}.csv"

        df_cv.write_csv(output_file)

        return output_file
    



import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
from typing import List, Dict, Any
import os


class GlobalAnalyzer:
    def __init__(self, results_dir: str = "results/"):
        self.results_dir = results_dir
        self.df_importances = None
        self.correlation_matrix = None
        self.mae_df = None

    def load_importance_dfs(self, models: List[str], perfs: List[str]) -> pl.DataFrame:
        """Charge les DataFrames d'importance et les aligne en colonne."""
        df_list = []

        for model in models:
            target_file = f"results_classif_logging2_{model}.csv"
            for perf in perfs:
                for performance in [
                    f"Mean_{perf}", f"Worst_{perf}", f"Best_{perf}", f"Std_{perf}"
                ]:
                    res = get_feature_importance_plot(model, perf, performance=performance, depth=10, cv=1)
                    res = res.with_columns(
                        (pl.lit(f"{model}_") + pl.col("column")).alias("column")
                    )
                    df_list.append(res)

        aligned = pl.concat(df_list, how="align_full")
        # Ajouter depth ici si tu veux le garder dans l’analyse ?
        self.df_importances = aligned
        return aligned

    def compute_kendall_correlation(self) -> pl.DataFrame:
        """Calcule la matrice de corrélation Kendall entre colonnes d’importance."""
        df = self.df_importances
        X = df.select(pl.exclude("depth", "column")).to_numpy()
        names = df["column"].to_list()

        K = [[kendalltau(X[i, :], X[j, :]).statistic for j in range(X.shape[0])] for i in range(X.shape[0])]
        self.correlation_matrix = pl.DataFrame(K, schema=names)

        return self.correlation_matrix

    def plot_kendall_heatmap(self, filename="corr.png"):
        """Plot la heatmap de corrélation Kendall."""
        corr = self.correlation_matrix
        plt.figure(figsize=(12, 12))
        sns.heatmap(
            corr.to_numpy(),
            xticklabels=corr.columns,
            yticklabels=corr.columns,
            cmap="coolwarm",
            linewidths=0.5
        )
        plt.title("Kendall Correlation of Feature Importances", fontsize=16)
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()
    @staticmethod
    def export_aggregated_importances(file):
        """Exporte les sommes et counts d’importances."""
        df = pl.read_csv(file)
        df = df.filter(pl.col("column").str.contains("Mean"))
        stem = os.path.splitext(file)[0]
        # transposed = df.sum().transpose(include_header=True, header_name="features")
        # transposed.sort("column_0").write_csv("sum_feature_importances.csv")

        # Sans depth et column
        transposed_no_meta = (
            df.sum()
            .drop(["depth", "column"])
            .transpose(include_header=True, header_name="features")
        )
        transposed_no_meta.sort("column_0").write_csv(f"sum_feature_importance_{stem}.csv")

        # Count non‑nulles
        df_none = df.with_columns(
            pl.all().map_elements(
                lambda x: None if x == 0.0 else x
            )
        )
        counts = df_none.count().transpose(include_header=True, header_name="features")
        counts.sort("column_0").write_csv(f"count_feature_importance{stem}.csv")

    def load_mae_results(self, models: List[str], perfs: List[str]) -> pl.DataFrame:
        """Relance LOO_train_DT et construit le DataFrame MAE."""
        rows = []
        for model in models:
            target_file = f"results_classif_logging2_{model}.csv"
            for perf in perfs:
                loo_results = LOO_train_DT(target_file, perf=perf, model=model)
                for _, _, mae_dict, dataset in loo_results:
                    # depth = 10, MAE = mae_dict[10]
                    values = mae_dict[10]
                    rows.append({
                        'model': model,
                        'metric': perf,
                        'Mean': values[0],
                        'worst': values[1],
                        'best': values[2],
                        'std': values[3],
                        'dataset': dataset,
                    })

        self.mae_df = pl.DataFrame(rows)
        return self.mae_df

    def top_mae_by_group(self, k=6, output_file="higher_MAE.csv"):
        """Retourne les top‑k MAEs par groupe (model, metric)."""
        if self.mae_df is None:
            raise ValueError("Run load_mae_results() first.")
        top = (
            self.mae_df
            .sort("Mean", descending=True)
            .group_by(["model", "metric"])
            .head(k)
        )
        top.write_csv(output_file)
        return top
