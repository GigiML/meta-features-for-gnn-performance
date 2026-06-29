import os
import optuna
import lightning as L
import torch
import click
import numpy as np
from src.gnn.model_hpo import GraphLitModel
from src.data.data import KfoldGraphDataModule
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger


# ── Constantes ────────────────────────────────────────────────────────────────

EPOCHS_TRIAL = 150  # Epochs par trial
N_TRIALS = 50  # Nombre de trials
DB_PATH = "sqlite:///optuna_studies/TUdataset.db"
NUM_SPLIT = 10


def objective(trial: optuna.Trial, name: str, net: str):
    L.seed_everything(42)

    # ── Parametre ────────────────────────────────────────────────────────────
    lr = trial.suggest_float("lr", 10**-4, 10**-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128])
    beta1 = trial.suggest_float("beta1", 0.5, 0.99)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 1e-2, log=False)
    laplacian_normalize = (
        trial.suggest_categorical("laplacian_normalize", [True, False]) 
        if net == "GCN" else False
    )


    balanced_acc_scores = []
    auprc_scores = []

    # ── Kfold loop ───────────────────────────────────────────────────────────
    for k in range(NUM_SPLIT):  # ← Votre syntaxe exacte
        print(f"Trial {trial.number} - Fold {k+1}/{NUM_SPLIT}")

        # ── Create module and model ────────────────────────────────────────
        dm = KfoldGraphDataModule(
            name=name, num_splits=NUM_SPLIT, batch_size=batch_size, k=k
        )
        dm.setup()

 
        model = GraphLitModel(
            net=net,
            in_channels=dm.input_dim,
            lr=lr,
            beta1=beta1,
            weight_decay=weight_decay,
            weights=dm.weights,
            normalize=laplacian_normalize,
        )
        # ── logger  ────────────────────────────────────────
        tb_logger = TensorBoardLogger("tb_logs/", name=f"{name}_{net}_t{trial.number}",log_graph=False, )
        csv_logger = CSVLogger("csv_logs", name=f"{name}_{net}_t{trial.number}")
        # mlf_logger = MLFlowLogger(
        #     experiment_name=f"{name}_K10Fold",
        #     run_name=f"t{trial.number}_f{k}",
        #     tracking_uri="file:./mlruns",
        # )
        # ── call back ────────────────────────────────────────
        early_stop_callback = EarlyStopping(
            monitor="val/loss", min_delta=0.01, patience=10, verbose=False, mode="min"
        )

        # ── Trainer ────────────────────────────────────────
        trainer = L.Trainer(
            max_epochs=EPOCHS_TRIAL,
            accelerator="gpu",
            devices=1,
            num_sanity_val_steps=0,
            callbacks=[early_stop_callback],
            logger=[tb_logger, csv_logger],
            enable_checkpointing=False,
            enable_progress_bar=False,
            log_every_n_steps=1000,
        )

        trainer.fit(model, datamodule=dm)
        fold_balanced_acc = trainer.callback_metrics.get("val/Balanced_Acc", 0.0)
        fold_auprc = trainer.callback_metrics.get("val/AUPRC", 0.0)

        balanced_acc_scores.append(fold_balanced_acc)
        auprc_scores.append(fold_auprc)
        trainer.test(model, datamodule=dm)

        del model, trainer, dm
        torch.cuda.empty_cache()

        mean_balanced_acc = np.mean(balanced_acc_scores)
        mean_auprc = np.mean(auprc_scores)

        
    return mean_balanced_acc, mean_auprc


@click.command()
@click.option(
    "--name",
    "-n",
    default="ER_MD",
    help="Dataset name (PROTEINS, AIDS, etc.)",
)
@click.option(
    "--net",
    "-n",
    default="GCN",
    help="Dataset name (PROTEINS, AIDS, etc.)",
)
def main(name: str, net: str):
    os.makedirs("optuna_studies", exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name=f"{name}_{net}",
        directions=["maximize", "maximize"],
        storage=DB_PATH,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    print(f"Study : {study.study_name} — {len(study.trials)} trials déjà réalisés")
    print(f"Lancement de {N_TRIALS} trials...\n")

    # À COMPLÉTER : lancer l'optimisation
    study.optimize(
        lambda trial: objective(trial, name, net),
        n_trials=N_TRIALS,
        show_progress_bar=True,
        gc_after_trial=True,  # Libere la memoire GPU
    )


if __name__ == "__main__":
    main()
