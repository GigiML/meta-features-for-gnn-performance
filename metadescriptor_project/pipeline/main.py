from src.pipeline.gbolt_metadescripteur_pipeline import run_pipeline
from src.analysis.summary_generation import generate_summary
from src.pipeline.save_pickle_csv_json import save_all_pickles_csv_json

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
    "ER_MD",
    "MUTAG",
    "PC-3",
    "PC-3H",
    "Yeast",
    "YeastH",
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

def main():
    run_pipeline(combined_datasets)
    save_all_pickles_csv_json()
    generate_summary(combined_datasets)

if __name__ == "__main__":
    main()

