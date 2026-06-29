import subprocess
import click
import os
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
models = ["GCN", "GIN", "GAT"]


VENV_PYTHON = "./.venv/bin/python"
PYTHONPATH = "../../."
OPTUNA_SCRIPT = "src/gnn/optuna_search.py"
@click.command()
@click.option(
    "--net", "-m",
    type=click.Choice(["GCN", "GIN", "GAT"], case_sensitive=False),
    default="GCN",
    help="Model architecture",
)
def main(net):
    env = os.environ.copy()
    env["PYTHONPATH"] = PYTHONPATH
    for name in combined_datasets:
        cmd = [VENV_PYTHON, OPTUNA_SCRIPT, "--net", net, "--name", name]
        # cmd = ["python", OPTUNA_SCRIPT, "--net", net, "--name", name]

        print(f"Lancement : {net} sur {name}")
        subprocess.run(cmd)

if __name__ == "__main__":
    main()
