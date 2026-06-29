import os
import polars as pl
from src.analysis.feature_importance import FeatureAnalysisConfig, ModelTrainer, PlotGenerator, _export_decision_trees, _train_models_comparison
from sklearn.model_selection import KFold, LeaveOneOut
import re
from src.pipeline.save_pickle_csv_json import save_all_pickles_csv_json
from src.analysis.summary_generation import generate_summary
import os 
import pickle
summary_file_column =  pl.read_csv("summary.csv").drop("Dataset").columns
def get_summary():

    save_all_pickles_csv_json() # generate the csv from the pickle 


    files = !ls SPMF/*.pkl 
    datasets = list()
    pattern = re.compile(r"SPMF/(.*?)_infos.pkl")

    for file in files:
        match = pattern.findall(file)
        datasets.append(match[0])

    generate_summary(datasets) # generate summary 


perfs = ["Balanced_Accuracy", "Accuracy", "F1_Score", "AUC"]

def get_performance(name_file, directory=".", perf="Accuracy"):
    file_path = os.path.join(directory, name_file)
    df = pl.read_csv(file_path)
    cols = ["Dataset", f"Mean_{perf}", f"Worst_{perf}", f"Best_{perf}", f"Std_{perf}"]
    return df.select(cols)

config = FeatureAnalysisConfig()

def LOO_train_DT(target_file="results_classif_logging1.csv", feature_file="summarywithnodelabels.csv", perf="Accuracy", model="GCN"):
    df = pl.read_csv(feature_file)
    df_perf = get_performance(target_file,perf=perf)
    X, y = pl.align_frames(df, df_perf, on="Dataset",how="inner")
    name_dataset = X["Dataset"]
    columns = X.drop("Dataset").columns
    columns_perf = y.drop("Dataset").columns
    X = X.drop("Dataset").to_numpy()
    y = y.drop("Dataset").to_numpy()


    cv_results = []
    #kf = KFold(n_splits=len(X), shuffle=True, random_state=42)
    loo = LeaveOneOut()
    for i,(train_index, test_index) in enumerate(loo.split(X)):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        res = _train_models_comparison(X_train, X_test, y_train, y_test, columns, columns_perf, config, None, "", None, True, cv=i, model_name=model)
        cv_results.append(res)
    return cv_results


def Kfold_train_DT(target_file="results_classif_logging1.csv", feature_file="summary.csv", perf="Accuracy", model="GCN"):
    df = pl.read_csv(feature_file)
    df_perf = get_performance(target_file,perf=perf)
    X, y = pl.align_frames(df, df_perf, on="Dataset",how="inner")
    name_dataset = X["Dataset"]
    columns = X.drop("Dataset").columns
    columns_perf = y.drop("Dataset").columns
    X = X.drop("Dataset").to_numpy()
    y = y.drop("Dataset").to_numpy()


    cv_results = []
    kf = KFold(n_splits=10, shuffle=False, random_state=42)
    for i,(train_index, test_index) in enumerate(kfoo.split(X)):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        res = _train_models_comparison(X_train, X_test, y_train, y_test, columns, columns_perf, config, None, "", None, True, cv=i, model_name=model)
        cv_results.append(res)
    return cv_results


def train_DT_on_all(target_file="results_classif_logging1.csv", feature_file="summary.csv", perf="Accuracy", model="GCN"):
    df = pl.read_csv(feature_file)
    df_perf = get_performance(target_file,perf=perf)
    X, y = pl.align_frames(df, df_perf, on="Dataset",how="inner")
    name_dataset = X["Dataset"]
    columns = X.drop("Dataset").columns
    columns_perf = y.drop("Dataset").columns
    X = X.drop("Dataset").to_numpy()
    y = y.drop("Dataset").to_numpy()


    cv_results = []
    res = _train_models_comparison(X,X, y, y, columns, columns_perf, config, None, "", None, True,cv=0, model_name=model)
    cv_results.append(res)
    return cv_results, name_dataset

def report_feature_importances(model_name, depth_max, perf="Accuracy", max_cv=0):
    columns_perf =  [f"Mean_{perf}", f"Worst_{perf}", f"Best_{perf}", f"Std_{perf}"]
    df = list()
    for depth in range(1, depth_max+1,1):
        for cv in range(max_cv):
            filename = f"models/multi_model_{model_name}_depth{depth}_Mean_{perf}_{cv}.pkl"
            with open(filename, 'rb') as f:
                model = pickle.load(f)
                for i,column in enumerate(columns_perf):
                    line = list()
                    line = line + [column, depth, cv]
                    line.extend(model.estimators_[i].feature_importances_)
                    df.append(line)
    df__ =pl.DataFrame(df, strict=False)
    columns =summary_file_column
    columns = ["column", "depth", "cv"] + columns
    df_transposed = df__.transpose()
    cols = df_transposed.columns[3:]
    df = df_transposed.with_columns(
    [pl.col(col).cast(pl.Float64, strict=False).alias(col) for col in cols]
    )
    df.columns = columns
    grouped_df = df.drop("cv").group_by(["column", "depth"], maintain_order=True).sum()
    grouped_df.write_csv(f"sum_feature_importances_{model_name}_{perf}_{max_cv}.csv")
    

import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

def get_feature_importance_plot(model, perf, performance, depth, cv=0):
    df = pl.read_csv(f"sum_feature_importances_{model}_{perf}_{cv}.csv")
    
    filtered = df.filter(
        (pl.col("column") == performance) & (pl.col("depth") == depth)
    )


    transposed = filtered.transpose(include_header=True, header_name="feature")
    

    transposed = transposed[2:].with_columns(pl.col("column_0").cast(pl.Float64))
    

    filtered_nonzero = transposed.filter(pl.col("column_0") > 0)

    filtered_sorted = filtered_nonzero.sort(pl.col("column_0"))

    features = filtered_sorted["feature"].to_list()
    importances = filtered_sorted["column_0"].to_list()
    save_path = f"figs/importance_{model}_depth{depth}_{performance}_cv{cv}.png"
    # Plot
    plt.figure(figsize=(25, 8))
    #plt.xticks(rotation=45)
    sns.barplot(x=importances, y=features)
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title(f"Feature Importance for model {model} - Performance: {performance}, Depth: {depth}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


    return filtered


    
#df=list()
#for model in ["GCN", "GIN", "SAGE", "GAT"]:
#    ...:     target_file = f"results_classif_logging2_{model}.csv"
#    ...:     for perf in ["Balanced_Accuracy", "Accuracy", "F1_Score", "AUC"]:
#    ...:         train_DT_on_all(target_file=target_file, model=model, perf=perf)
#    ...:         report_feature_importances(model_name=model, depth_max=10, max_cv=1, perf=perf)
#    ...:         for performance in  [ f"Mean_{perf}", f"Worst_{perf}", f"Best_{perf}", f"Std_{perf}"]:
#    ...:           res = get_feature_importance_plot(model, perf,performance=performance, depth=10, cv=1)
                    # res = get_feature_importance_plot(model, perf,performance=performance, depth=10, cv=1)
                    #res = res.with_columns(
                    #             (pl.lit(f"{model}_") + pl.col("column")).alias("column")
                    #                 )
                    #res["column"].replace(res["column"], f"{model}_{res["column"]}")
                    # df.append(res)
# df_align = pl.concat(df, how="align_full")
# df_mean = df_align.filter(pl.col("column").str.contains("Mean"))
# df_None_mean = df_mean.with_columns(
#     ...:     pl.all().map_elements(
#     ...:         lambda x: None if x == 0.0 else x
#     ...:     )
#     ...: )

# col = df_None_mean.drop(['depth', 'column']).count().transpose(include_header=True, header_name="
#        ⋮ features").sort("column_0")["features"][-15:]

# df_mean.select(["column"] + col.to_list()[::-1]).write_csv("top_15_fetaure_importance_5_layer.csv")
