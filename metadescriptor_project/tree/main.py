import numpy as np
from src.tree.dataset_dtr import Dataset
from src.tree.Analyzer import DecisionTreeAnalyzer
from src.tree.Analyzer import GlobalAnalyzer
import numpy as np
from itertools import product
from src.tree.dataset_dtr import Dataset
from src.tree.Analyzer import DecisionTreeAnalyzer
import click
import pandas as pd

flag_names = ["family", "source"]


def run_dt(
    family,
    source,
    model="GCN",
    ablation=True,
    summary_file="summary.csv",
    target_file="results_classif_logging5_normalizeGCN.csv",
    perf="Accuracy",
    depth_max=10,
    cv_type="all",
):
    # kwargs = dict(zip(flag_names, flags_vals))

    dt = DecisionTreeAnalyzer(
        model=model,
        summary_file=summary_file,
        target_file=target_file,
        perf=perf,
        depth_max=depth_max,
        cv_type=cv_type,
        family=family,
        source=source,
        ablation=ablation,
    )
    if cv_type == "kfold":
        max_cv = 9
    elif cv_type == "loo":
        max_cv = 64
    else:
        max_cv = 1
    if ablation:
        columns = Dataset.get_feature_ablation(family, source)
    else:
        columns = Dataset.get_one_combination(family, source)
    cv_results, mae, name_dataset = dt.train()
    output_file = dt.report_feature_importances(
        max_cv=max_cv, summary_file_column=columns
    )
    dt.save_cv_results(cv_results, perf)

    # GlobalAnalyzer.export_aggregated_importances(output_file)
    if cv_type == "loo":
        pd.DataFrame({"name_dataset": name_dataset, "mae": mae}).sort_values(
            by="mae"
        ).to_csv(f"MAE_LOO_{model}_{perf}.csv")

    return {
        "kwargs": f"{family, source}",
        "cv_results": cv_results,
        "name_dataset": name_dataset,
        "mae": mae,
    }


@click.command()
@click.option("--model", default="GCN", show_default=True, help="Model name.")
@click.option(
    "--perf", default="Accuracy", show_default=True, help="Performance metric."
)
@click.option("--cv_type", default="all", show_default=True, help="all or loo or kfold")
def main(model, perf, cv_type):
    target_file = f"{model}.csv"
    # all
    print(
        run_dt(
            None,
            None,
            cv_type=cv_type,
            model=model,
            perf=perf,
            target_file=target_file,
            ablation=True,
        )
    )
    families = ["F1", "F2", "F3", "F4"]
    source = ["NORMAL", "GSPAN"]
    flags_configs = product(families, source)
    if cv_type == "all":
        # ablation studies
        # for family, source in flags_configs:
        #     run_dt(
        #         cv_type=cv_type,
        #         family=family,
        #         source=source,
        #         model=model,
        #         perf=perf,
        #         target_file=target_file,
        #     )

        # One family per one
        for family, source in flags_configs:
            run_dt(
                cv_type=cv_type,
                family=family,
                source=source,
                ablation=False,
                model=model,
                perf=perf,
                target_file=target_file,
            )


if __name__ == "__main__":
    main()
