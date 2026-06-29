import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error as mae 
from sklearn.inspection import permutation_importance
import pydotplus
from scipy.stats import pearsonr
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
import os
import pickle


class FeatureAnalysisConfig:
    """Configuration class for feature analysis parameters."""
    
    def __init__(self, max_depth=7, min_impurity_decrease=0.0001, min_samples_leaf=2,
                 nb_folds=10, nb_top=5, random_state=42):
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.min_samples_leaf = min_samples_leaf
        self.nb_folds = nb_folds
        self.nb_top = nb_top
        self.random_state = random_state


class DataLoader:
    """Class for loading and preprocessing data."""
    
    @staticmethod
    def load_data(summary_file, accuracy_file):
        """Load and preprocess summary and accuracy data."""
        summary_pd = pd.read_csv(summary_file, index_col=0, dtype={"Dataset": str})
        summary_pd = summary_pd.apply(pd.to_numeric, errors='coerce')
        
        acc_pd = pd.read_csv(accuracy_file, index_col=0, dtype={"Dataset": str})
        acc_pd = acc_pd.apply(pd.to_numeric, errors='coerce')
        
        columns_pd_data = summary_pd.columns
        columns_pd_acc = acc_pd.columns
        
        pd_data = pd.concat([summary_pd, acc_pd], axis=1)
        pd_data = pd_data.dropna(subset=columns_pd_acc).dropna()
        
        X = pd_data.drop(columns=columns_pd_acc).values
        y = pd_data[columns_pd_acc].to_numpy()
        
        return summary_pd, acc_pd, pd_data, X, y, columns_pd_data, columns_pd_acc


class CorrelationAnalyzer:
    """Class for correlation analysis and clustering."""
    
    @staticmethod
    def compute_correlation_matrix(df):
        """Compute correlation matrix with proper handling of NaN values."""
        df = df.fillna(0)
        cols = df.columns.tolist()
        n = len(cols)
        corr_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
        
        for i in tqdm(range(n), desc="Calcul des corrélations"):
            col_i = cols[i]
            xi = df[col_i]
            
            for j in range(i, n):
                col_j = cols[j]
                xj = df[col_j]
                
                mask = xi.notna() & xj.notna()
                if mask.sum() < 2:
                    corr = np.nan
                else:
                    corr, _ = pearsonr(xi[mask], xj[mask])
                
                corr_matrix.at[col_i, col_j] = corr
                corr_matrix.at[col_j, col_i] = corr
        
        return corr_matrix
    
    @staticmethod
    def perform_clustering(corr_matrix, threshold=0.25):
        """Perform hierarchical clustering on correlation matrix."""
        corr_matrix = corr_matrix.fillna(0)
        distance_matrix = 1 - corr_matrix.abs()
        np.fill_diagonal(distance_matrix.values, 0)
        condensed_dist = squareform(distance_matrix.values)
        
        Z = linkage(condensed_dist, method='average')
        clusters = fcluster(Z, t=threshold, criterion='distance')
        
        col_names = corr_matrix.columns.tolist()
        group_dict = {}
        for col, group_id in zip(col_names, clusters):
            group_dict.setdefault(group_id, []).append(col)
        
        group_dict = {gid: cols for gid, cols in group_dict.items() if len(cols) > 1}
        
        return group_dict, Z


class ModelTrainer:
    """Class for training decision tree models."""
    
    def __init__(self, config):
        self.config = config
    
    def create_decision_tree(self, free_tree=True):
        """Create a decision tree with specified parameters."""
        return DecisionTreeRegressor(
            max_depth=self.config.max_depth,
            random_state=self.config.random_state,
            min_impurity_decrease=self.config.min_impurity_decrease,
            min_samples_leaf=self.config.min_samples_leaf
        )
    
    def train_multioutput_model(self, X, y, free_tree=True):
        """Train a multioutput decision tree model."""
        multi_model = MultiOutputRegressor(self.create_decision_tree(free_tree))
        multi_model.fit(X, y)
        return multi_model


class FileManager:
    """Class for managing file paths and directories."""
    
    @staticmethod
    def get_output_path(base_name, reduce=False, my_columns=False, threshold=None, extension=""):
        """Generate output file path based on analysis type."""
        if reduce:
            return f"src/tmp/{base_name}_reduced_{threshold}{extension}"
        elif my_columns:
            return f"src/tmp/{base_name}_my_columns_{threshold}{extension}"
        else:
            return f"src/tmp/{base_name}{extension}"
    
    @staticmethod
    def ensure_tmp_dir():
        """Ensure tmp directory exists."""
        os.makedirs("src/tmp", exist_ok=True)


class PlotGenerator:
    """Class for generating plots and visualizations."""
    
    @staticmethod
    def save_dendrogram(Z, col_names, reduce=False, my_columns=False, threshold=None):
        """Save correlation dendrogram."""
        plt.figure(figsize=(12, 6))
        dendrogram(Z, labels=col_names, leaf_rotation=90)
        
        if reduce:
            plt.title(f"Dendrogramme des corrélations (réduit, seuil={threshold})")
            filename = f"src/tmp/dendrogram_correlation_reduced_{threshold}.png"
        elif my_columns:
            plt.title("Dendrogramme des corrélations (mes colonnes)")
            filename = f"src/tmp/dendrogram_correlation_my_columns_{threshold}.png"
        else:
            plt.title("Dendrogramme des corrélations")
            filename = "src/tmp/dendrogram_correlation.png"
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
    
    @staticmethod
    def save_r2_comparison(manual_scores, multi_scores, dims_names, max_depth,perf="Accuracy",
                          reduce=False, my_columns=False, threshold=None, cv=0, family="F1", source="NORMAL", ablation=True):
        """Save R² comparison plot."""
        dim = len(manual_scores)
        x = np.arange(dim)
        width = 0.35
        
        plt.figure(figsize=(10, 5))
        plt.bar(x - width/2, manual_scores, width, label='Boucle manuelle', color='skyblue')
        plt.bar(x + width/2, multi_scores, width, label='MultiOutputRegressor', color='salmon')
        plt.xticks(x, dims_names, rotation=45)
        plt.ylabel("Score R²")
        plt.title(f"Comparaison des scores R² par dimension pour max_depth={max_depth}")
        plt.legend()
        plt.tight_layout()
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        filename = FileManager.get_output_path(
            f"Comparaison_scores_R2_max_depth_{perf}_{max_depth}_{source}_{family}_{ablation}_cv_{cv}",
            reduce, my_columns, threshold, ".png"
        )
        plt.savefig(filename)
        plt.close()
    
    @staticmethod
    def save_importance_plot(top_vars, top_importances, max_depth,
                           reduce=False, my_columns=False, threshold=None):
        """Save feature importance plot."""
        plt.figure(figsize=(12, 8))
        plt.bar(top_vars, top_importances, alpha=0.7, color='blue')
        plt.xlabel('Variables')
        plt.ylabel('Importance (%)')
        plt.title(f'Importance des variables pour max_depth={max_depth}')
        plt.xticks(rotation=90)
        plt.tight_layout()
        
        filename = FileManager.get_output_path(
            f"importance_variables_max_depth_{max_depth}",
            reduce, my_columns, threshold, ".png"
        )
        plt.savefig(filename)
        plt.close()
    
    @staticmethod
    def save_multilabel_importance_plot(df_pivot, max_depth,
                                      reduce=False, my_columns=False, threshold=None):
        """Save multilabel importance plot."""
        df_pivot.plot(kind="bar", figsize=(14, 8), width=0.8, colormap="viridis", alpha=0.8)
        plt.xlabel('Variables')
        plt.ylabel('Importance (%)')
        plt.title(f'Importance des variables pour chaque axe (max_depth={max_depth})')
        plt.xticks(rotation=90)
        plt.legend()
        plt.tight_layout()
        
        filename = FileManager.get_output_path(
            f"importance_variables_multilabel_{max_depth}",
            reduce, my_columns, threshold, ".png"
        )
        plt.savefig(filename)
        plt.close()


class DecisionTreeExporter:
    """Class for exporting decision trees."""
    
    @staticmethod
    def export_tree(estimator, columns_pd_data, target_name, max_depth, precision,
                   reduce=False, my_columns=False,model_name="", threshold=None, cv=0,source="NORMAL", family="F1", ablation=True):
        """Export decision tree to PNG."""
        leaf_purity = estimator.tree_.impurity[estimator.tree_.children_left == -1]
        leaf_sizes = estimator.tree_.n_node_samples[estimator.tree_.children_left == -1]
        
        dot_data = export_graphviz(estimator, out_file=None,
                                  feature_names=columns_pd_data,
                                  filled=True, rounded=True,
                                  special_characters=True)
        dot_graph = pydotplus.graph_from_dot_data(dot_data)
        
        if reduce:
            label = f"Decision Tree {target_name} - reduce {threshold}\\nmax_depth={max_depth} - PRECISION : {precision:.2f}\\nIMPURITY : {(np.mean(leaf_purity) / np.sum(leaf_purity) * 100):.2f} - SIZES {np.mean(leaf_sizes):.2f}"
            filename = f"src/tmp/decision_tree_{model_name}_{target_name}_depth_{max_depth}_reduced_{threshold}.png"
        elif my_columns:
            label = f"Decision Tree {target_name} - my columns\\nmax_depth={max_depth} - PRECISION : {precision:.2f}\\nIMPURITY : {(np.mean(leaf_purity) / np.sum(leaf_purity) * 100):.2f} - SIZES {np.mean(leaf_sizes):.2f}"
            filename = f"src/tmp/decision_tree_{model_name}_{target_name}_depth_{max_depth}_my_columns_{threshold}.png"
        else:
            label = f"Decision Tree {target_name}\\nmax_depth={max_depth} - PRECISION : {precision:.2f}\\nIMPURITY : {(np.mean(leaf_purity) / np.sum(leaf_purity) * 100):.2f} - SIZES {np.mean(leaf_sizes):.2f}"
            filename = f"src/tmp/decision_tree_{model_name}_{target_name}_depth_{max_depth}_{source}_{family}_{ablation}cv_{cv}.png"
        
        dot_graph.set_label(label)
        dot_graph.set_labelloc("t")
        dot_graph.write_png(filename)


class AdvancedAnalyzer:
    """Class for advanced feature analysis using iterative decision trees."""
    
    @staticmethod
    def run_alternative_analysis(summary_file="results/summaries/summary.csv", accuracy_file="results/accuracy.csv", 
                                reduce=False, threshold=0.9):
        """
        Alternative feature analysis using iterative decision trees to identify 
        correlated feature groups based on root features.
        """
        print("Running alternative feature analysis..., threshold:", threshold)

        # Load data
        summary_pd, acc_pd, pd_data, X, y, columns_pd_data, columns_pd_acc = DataLoader.load_data(
            summary_file, accuracy_file
        )
        
        # Use only 'Mean' for this analysis
        if "Mean" not in acc_pd.columns:
            raise ValueError("La colonne 'Mean' est absente du fichier d'accuracy.")

        meta_desc = columns_pd_data.tolist()
        X = pd_data[meta_desc].values
        y = pd_data["Mean"].to_numpy()

        groups = {}
        used = set()

        # Decision tree parameters
        config = FeatureAnalysisConfig()
        
        # Ensure output directory exists
        FileManager.ensure_tmp_dir()
        
        # Output file
        output_path = FileManager.get_output_path(
            "correlation_groups_alternative", reduce, False, threshold, ".txt"
        )
        
        with open(output_path, "w") as f:
            f.write("")

        # Main loop
        while meta_desc:
            print(f"\n[Itération] Méta-descripteurs restants : {len(meta_desc)}")

            trainer = ModelTrainer(config)
            model = trainer.create_decision_tree()
            model.fit(X, y)

            root_index = model.tree_.feature[0]
            if root_index >= len(meta_desc):
                print("Index de racine invalide, arrêt.")
                break
            
            try:
                root_meta = meta_desc[root_index]
            except IndexError:
                print(f"Index de racine invalide, arrêt. Racine : {root_index}, META_DESC length : {len(meta_desc)}")
                break
                
            print(f"Racine sélectionnée : {root_meta}")

            # Permutation importance BEFORE removal
            pi = permutation_importance(model, X, y, n_repeats=5, random_state=42)
            accuracy = pi.importances_mean[root_index]

            # Calculate correlation only for root_meta
            correlated = []
            x_root = pd_data[root_meta]
            for other in meta_desc:
                if other == root_meta:
                    continue
                x_other = pd_data[other]
                mask = x_root.notna() & x_other.notna()
                if mask.sum() < 2:
                    continue
                corr, _ = pearsonr(x_root[mask], x_other[mask])
                if abs(corr) >= threshold:
                    correlated.append(other)

            # Build group
            current_group = [root_meta] + correlated
            groups[root_meta] = current_group

            with open(output_path, "a") as f:
                f.write(f"Groupe {root_meta} : {', '.join(current_group)} - Permutation importance : {accuracy:.4f}\n")

            print(f"Groupe {root_meta} : {current_group} - Permutation importance : {accuracy:.4f}")

            # Update: remove group columns from meta_desc and X
            indices_to_delete = [meta_desc.index(col) for col in current_group if col in meta_desc]
            X = np.delete(X, indices_to_delete, axis=1)

            for col in current_group:
                if col in meta_desc:
                    meta_desc.remove(col)
                    used.add(col)

        print("\nAnalyse terminée.")
        return groups


def run_feature_analysis(summary_file="results/summaries/summary.csv", accuracy_file="results/accuracy.csv", 
                        reduce=False, threshold=0.9, my_columns=False, free_tree=True, 
                        min_impurity_decrease=0.0001, min_samples_leaf=2):
    """
    Main function to run feature analysis including correlation analysis,
    importance calculation, and model training.
    """
    print(f"Running feature analysis..., threshold: {threshold}")
    
    # Initialize configuration
    config = FeatureAnalysisConfig(
        min_impurity_decrease=min_impurity_decrease,
        min_samples_leaf=min_samples_leaf
    )
    
    # Ensure output directory exists
    FileManager.ensure_tmp_dir()
    
    # Load data
    summary_pd, acc_pd, pd_data, X, y, columns_pd_data, columns_pd_acc = DataLoader.load_data(
        summary_file, accuracy_file
    )
    
    # Compute correlation matrix
    df = pd.read_csv(summary_file, index_col=0, dtype={"Dataset": str})
    corr_matrix = CorrelationAnalyzer.compute_correlation_matrix(df)
    
    # Save correlation matrix
    corr_filename = FileManager.get_output_path(
        "correlation_matrix_column_by_column", reduce, my_columns, threshold, ".csv"
    )
    corr_matrix.to_csv(corr_filename)
    
    # Perform clustering
    group_dict, Z = CorrelationAnalyzer.perform_clustering(corr_matrix, threshold=0.25)
    
    # Save clustering results
    _save_clustering_results(group_dict, reduce, my_columns, threshold)
    
    # Save dendrogram
    col_names = corr_matrix.columns.tolist()
    PlotGenerator.save_dendrogram(Z, col_names, reduce, my_columns, threshold)
    
    # Calculate feature importance rankings
    _calculate_feature_importance_rankings(
        X, y, col_names, config, reduce, my_columns, threshold, free_tree
    )
    
    # Train models with different depths and generate comparisons
    _train_models_comparison(
        X, y, columns_pd_data, columns_pd_acc, config, 
        reduce, my_columns, threshold, free_tree
    )


def _save_clustering_results(group_dict, reduce, my_columns, threshold):
    """Save clustering results to file."""
    for gid, cols in group_dict.items():
        print(f"Groupe {gid} : {cols}")
    
    filename = FileManager.get_output_path(
        "correlation_groups", reduce, my_columns, threshold, ".txt"
    )
    
    with open(filename, "w") as f:
        for gid, cols in group_dict.items():
            f.write(f"Groupe {gid} : {', '.join(cols)}\n")


def _calculate_feature_importance_rankings(X, y, col_names, config, reduce, my_columns, threshold, free_tree):
    """Calculate feature importance rankings for each target dimension."""
    columns_importance = ["Mean", "Best", "Worst", "std"]
    len_ligne_importance = len(col_names)
    importance_pd = pd.DataFrame(index=col_names, columns=columns_importance)
    
    print(f"X = {X.shape}, col = {len(col_names)}")
    
    # Create copies of X for each dimension
    Xs = [pd.DataFrame(X.copy(), columns=col_names) for _ in range(len(columns_importance))]
    
    # Calculate importance rankings for each feature
    for i in range(len_ligne_importance):
        print(f"\nNuméro = {i} / {len_ligne_importance}\n")
        
        trainer = ModelTrainer(config)
        
        for n, Xx in enumerate(Xs):
            col_names_current = Xx.columns
            X_used = Xx.to_numpy()
            
            # Create and train model
            model = trainer.create_decision_tree(free_tree)
            model.fit(X_used, y[:, n])
            
            # Calculate permutation importance
            p_i = permutation_importance(model, X_used, y[:, n], n_repeats=5, random_state=42)
            sorted_idx = np.argsort(p_i.importances_mean)[::-1]
            metadescr = col_names_current[sorted_idx[0]]
            
            # Store ranking
            importance_pd.loc[metadescr, columns_importance[n]] = i + 1
            
            # Remove most important feature for next iteration
            Xx = Xx.drop(columns=metadescr)
            Xs[n] = Xx
            
            print(f"Variable la plus importante pour {columns_importance[n]} : {metadescr} "
                  f"avec importance {p_i.importances_mean[sorted_idx[0]]:.4f}")
    
    # Save importance rankings
    filename = FileManager.get_output_path(
        "importance_variables", reduce, my_columns, threshold, ".csv"
    )
    importance_pd.to_csv(filename)


def _train_models_comparison(X_train,X_test, y_train, y_test, columns_pd_data, columns_pd_acc, config, 
                           reduce, my_columns, threshold, free_tree, cv=0, model_name="", source="NORMAL", family="F1", ablation=True):
    """Train model with depth = 10 and generate  plot."""
    max_depth = 10
    dim = y_train.shape[1]
    print(len(X_test), "hhh")
    resultats = {}

    print(f"\n\nRunning with max_depth: {max_depth}\n\n")
    
    # Update config for current depth
    current_config = FeatureAnalysisConfig(
        max_depth=max_depth,
        min_impurity_decrease=config.min_impurity_decrease,
        min_samples_leaf=config.min_samples_leaf
    )
    trainer = ModelTrainer(current_config)
    
    manual_scores = []
    for dim_idx in range(dim):
        model = trainer.create_decision_tree(free_tree)
        model.fit(X_train, y_train[:, dim_idx])
        score = model.score(X_test, y_test[:, dim_idx])
        manual_scores.append(score)
    #Train multioutput model
    multi_model = trainer.train_multioutput_model(X_train, y_train, free_tree)
    multi_scores = [est.score(X_test, y_test[:, i]) for i, est in enumerate(multi_model.estimators_)]
    multi_scores = np.array(multi_scores)
    resultats[max_depth] = multi_scores
    multi_filename = f"models/multi_model_{model_name}_depth{max_depth}_{"_".join(columns_pd_acc)}_{source}_{family}_{ablation}_{cv}.pkl"
    with open(multi_filename, 'wb') as f:
        pickle.dump(multi_model, f)
    print(y_test)
    print("MAE",mae(y_test, multi_model.predict(X_test)))
    #Generate comparison plot
    print("_".join(columns_pd_acc))
    dims_names = columns_pd_acc if len(columns_pd_acc) == dim else [f'dim_{i}' for i in range(dim)]
    PlotGenerator.save_r2_comparison(
        manual_scores, manual_scores, dims_names, max_depth, str("_".join(columns_pd_acc)), 
        reduce, my_columns, threshold, cv= cv, family=family, source=source, ablation=ablation
    )

    
    # Export decision trees for this depth
    _export_decision_trees(
        multi_model, columns_pd_data, columns_pd_acc, X_test, y_test, max_depth,
        reduce, my_columns, model_name, threshold, cv = cv
    )
    return  resultats, mae(y_test, multi_model.predict(X_test))


def _calculate_permutation_importance(X, y, columns_pd_data, trainer, free_tree, resultats, dim):
    """Calculate permutation importance for current model configuration."""
    for dim_idx in range(dim):
        model = trainer.create_decision_tree(free_tree)
        model.fit(X, y[:, dim_idx])
        result = permutation_importance(model, X, y[:, dim_idx], n_repeats=5, random_state=42)
        mean_importance = result.importances_mean
        
        for m, importance in enumerate(mean_importance):
            resultats[columns_pd_data[m]] = importance + resultats.get(columns_pd_data[m], 0)


def _export_decision_trees(multi_model, columns_pd_data, columns_pd_acc, X, y, max_depth,
                          reduce, my_columns, model_name, threshold, cv, source="NORMAL", family="F1", ablation=True):
    """Export decision trees for each estimator."""
    for i, estimator in enumerate(multi_model.estimators_):
        precision = estimator.score(X, y[:, i])
        target_name = columns_pd_acc[i] if i < len(columns_pd_acc) else f"target_{i}"
        
        DecisionTreeExporter.export_tree(
            estimator, columns_pd_data, target_name, max_depth, precision,
            reduce, my_columns,model_name, threshold, cv,source, family, ablation
        )


def _save_final_importance_results(resultats, reduce, my_columns, threshold):
    """Save final aggregated importance results."""
    resultat_sum = sum(resultats.values())
    if resultat_sum == 0:
        print("Aucune importance calculée. Vérifiez les données.")
        return
    
    # Normalize to percentages
    resultats = {k: v / resultat_sum * 100 for k, v in resultats.items()}
    
    # Sort by importance and get top features
    sorted_resultats = sorted(resultats.items(), key=lambda x: x[1], reverse=True)
    top_vars = [item[0] for item in sorted_resultats[:10]]  # Top 10
    top_importances = [item[1] for item in sorted_resultats[:10]]
    
    # Generate importance plot
    PlotGenerator.save_importance_plot(
        top_vars, top_importances, 10, 
        reduce, my_columns, threshold
    )


# Convenience function for alternative analysis
def run_feature_analysis_2(summary_file="results/summaries/summary.csv", accuracy_file="results/accuracy.csv", 
                          reduce=False, threshold=0.9):
    """Wrapper function for alternative feature analysis."""
    return AdvancedAnalyzer.run_alternative_analysis(summary_file, accuracy_file, reduce, threshold)
