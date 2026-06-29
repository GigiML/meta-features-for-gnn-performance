import os
import pandas as pd
import optuna

combined_datasets = sorted([
    "Tox21_AhR_training",
    "Tox21_AhR_evaluation",
    "Tox21_AR_training",
    "Tox21_AR-LBD_training",
    "Tox21_ARE_training",
    "Tox21_aromatase_training",
    "Tox21_ATAD5_training",
    "Tox21_ER_training",
    "Tox21_ER-LBD_training",
    "Tox21_HSE_training",
    "Tox21_MMP_training",
    #"BZR",
    #"COX2",
    #"DHFR",
    #"ER_MD",
    "MUTAG",
    "PC-3",
    "PC-3H",
    "Yeast",
    #"YeastH",
    "UACC257H",
    "NCI-H23",
    "NCI-H23H",
    "Tox21_AR_evaluation",
    "Tox21_AR-LBD_evaluation",
    "Tox21_ER_testing",
    "Tox21_PPAR-gamma_testing",
    "UACC257",
    "Tox21_aromatase_testing",
    "Tox21_ER-LBD_evaluation",
    "Tox21_ER-LBD_testing",
    "OVCAR-8",
    "Tox21_ATAD5_evaluation",
    "SW-620H",
    "SW-620",
    "Tox21_MMP_testing",
    "Tox21_MMP_evaluation",
    "MCF-7H",
    "MCF-7",
    "Tox21_p53_training",
    "SN12C",
    "PTC_FM",
     "PTC_FR",
      "PTC_MR",
    "MOLT-4H",
    "AIDS",
    "Tox21_ARE_evaluation",
    "OVCAR-8H",
    "SF-295H",
    "Tox21_PPAR-gamma_training",
    "Tox21_PPAR-gamma_evaluation",
    "SN12CH",
    "Tox21_AR_testing",
    "SF-295",
    "P388",
    "Tox21_p53_evaluation",
    "Tox21_AR-LBD_testing",
    "Tox21_AhR_testing",
    "MOLT-4",
    "Tox21_ER_evaluation",
    "Tox21_HSE_testing",
    "Tox21_aromatase_evaluation",
    "Mutagenicity",
    "P388H",
    "Tox21_ATAD5_testing",
    "PTC_MM",
    "Tox21_HSE_evaluation",
    "Tox21_p53_testing",
    "Tox21_ARE_testing"
])

def get_value(df, col):
    return df[col].dropna().iloc[-1]



# def read_trial_info(dataset_name, model_name, study_name, storage="sqlite:///optuna_studies/TUdataset.db"    , log_root="csv_logs", val_cols="test/Accuracy"):
#     study = optuna.load_study(study_name=study_name, storage=storage)
#     best_trial = study.best_trials[0]
#     metrics = dict()
#     def get_values(val_col):
#         val = list()
#         for i in range(10):
#             metrics_csv = os.path.join(log_root, f"{dataset_name}_t{best_trial.number}_f{i}/version_0/metrics.csv")

#             if not os.path.exists(metrics_csv):
#                 raise FileNotFoundError(f"metrics.csv introuvable: {metrics_csv}")

#             df = pd.read_csv(metrics_csv)
#             val.append(get_values(df, val_col))
#         return sum(val)/10
    
#     for val_col in val_cols:
#         metrics[val_col]=get_value(val_col)
#     row = {
#         "dataset_name": dataset_name,
#         "model_name": model_name,
#         **metrics
#     }

#     return row




def read_trial_info(dataset_name, model_name, study_name, storage="sqlite:///optuna_studies/TUdataset.db"    , log_root="csv_logs", val_cols="test/Accuracy"):
    study = optuna.load_study(study_name=study_name, storage=storage)
    best_trial = study.best_trials[0]
    metrics = dict()
    def get_values(val_col):
        val = list()
        for i in range(10):
            metrics_csv = os.path.join(log_root, f"{dataset_name}_t{best_trial.number}/version_{i}/metrics.csv")

            if not os.path.exists(metrics_csv):
                raise FileNotFoundError(f"metrics.csv introuvable: {metrics_csv}")

            df = pd.read_csv(metrics_csv)
            val.append(get_values(df, val_col))
        return sum(val)/10
    
    for val_col in val_cols:
        metrics[val_col]=get_values(val_col)
    row = {
        "dataset_name": dataset_name,
        "model_name": model_name,
        **metrics
    }

    return row
    

val_cols=["test/AUPRC","test/AUROC","test/Accuracy","test/Balanced_Acc","test/F1"]
pd.DataFrame([ read_trial_info(dataset, "GAT",f"{dataset}_GAT", val_cols=val_cols)  for dataset in combined_datasets]).to_csv("GAT.csv")

