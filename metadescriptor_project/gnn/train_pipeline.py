import os
import csv
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from src.gnn.model import GNN, GIN, GAT, GraphSAGE, train, test, k_fold
from dotenv import load_dotenv
from lightning.pytorch.loggers import CSVLogger
import copy
import warnings
import numpy as np
from torch_geometric import seed_everything
from torch_geometric.transforms import NormalizeFeatures
load_dotenv()
seed_everything(42)

GNN_dict = {
    "GCN": GNN,   
    "GIN": GIN,
    "GAT": GAT,
    "SAGE": GraphSAGE, 
}
def train_gnn(name, model_name="GCN"):
    ## LOAD LE DATASET
    dataset_dir = os.getenv("root","data/TUDataset")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    dataset_path = os.path.join(dataset_dir, name)


    #if not os.path.isdir(dataset_path):
    #    continue
    transform = NormalizeFeatures()

    dataset = TUDataset(root=dataset_dir, name=name,transform=transform)
    #if dataset.num_classes < 2:
    #    continue
    
    unique, counts = np.unique(dataset.y, return_counts=True)

    # Pour weighted cross entropy
    total = len(dataset.y)
    class_weights = total / (len(unique) * counts.astype(float))
    csv_filename = f"results_classif_logging5_noramlize{model_name}.csv"
    file_exists = os.path.exists(csv_filename)
    fieldnames = [
    'Dataset', 
    'Mean_Accuracy', 'Best_Accuracy', 'Worst_Accuracy', 'Std_Accuracy',
    'Mean_Balanced_Accuracy', 'Best_Balanced_Accuracy', 'Worst_Balanced_Accuracy', 'Std_Balanced_Accuracy',
    'Mean_F1_Score', 'Best_F1_Score', 'Worst_F1_Score', 'Std_F1_Score',
    'Mean_AUC', 'Best_AUC', 'Worst_AUC', 'Std_AUC',
    'Confusion_Matrix',  
    'Classification_Report'  
]
    if not file_exists:
        with open(csv_filename, mode='w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()


    dataset = dataset.shuffle()
    folds =  10
    accs = []
    hyperparams = {
    "lr": 0.001,
    "batch_size": 64,
    "epochs": 200,
     "weight_decay": 5e-4,
    "patience": 10,
    }
    csv_logger = CSVLogger(save_dir=f"logs/{model_name}", name=f"csv_logs_{name}", version=None)
    csv_logger.log_hyperparams(hyperparams)
    accs, bal_accs, f1s, aucs = [], [], [], []
    confusion_matrices = []
    classification_reports = []
    for fold, (train_idx, test_idx,
               val_idx) in enumerate(zip(*k_fold(dataset, folds))):

        best_model = None
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]

        train_loader = DataLoader(train_dataset, batch_size=hyperparams["batch_size"], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=hyperparams["batch_size"], shuffle=False)
        val_loader =  DataLoader(val_dataset, batch_size=hyperparams["batch_size"], shuffle=False)

        best_model = GNN_dict[model_name](dataset.num_node_features, 32, dataset.num_classes, num_layers=5,).to(device)
        model =  GNN_dict[model_name](dataset.num_node_features, 32, dataset.num_classes,num_layers=5).to(device)
        optimizer = Adam(model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["weight_decay"])
        criterion = CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))

        
        best_val_metrics = None
        best_acc = 0
        patience = hyperparams['patience']
        counter = 0
        for epoch in range(hyperparams["epochs"]):
            train_loss = train(model, train_loader, optimizer, criterion, device)
            
            val_metrics = test(val_loader, model, device)
            val_acc = val_metrics["accuracy"]
            # Ajouter epoch, fold, loss dans le dict des métriques à logger
            metrics_to_log = val_metrics.copy()
            metrics_to_log.update({
                "epoch": epoch,
                "fold": fold,
                "train_loss": train_loss
            })

            csv_logger.log_metrics(metrics_to_log, step=fold * hyperparams["epochs"] + epoch)
            

            if val_acc > best_acc:
                    best_acc = val_acc
                    best_val_metrics = val_metrics
                    best_model.load_state_dict(model.state_dict())
                    counter = 0
            else:
                counter += 1

            if counter >= patience:
                break
        test_metrics = test(test_loader, model, device)
        test_metrics.update({"fold": fold, "test": "test"})
        csv_logger.log_metrics(metrics_to_log, step=fold * hyperparams["epochs"] + epoch+1)
        csv_logger.save()
        accs.append(test_metrics['accuracy'])
        bal_accs.append(test_metrics['balanced_accuracy'])
        f1s.append(test_metrics['f1_score'])
        aucs.append(test_metrics['auc'])
        confusion_matrices.append(str(test_metrics['confusion_matrix']))  # converti en string
        classification_reports.append(str(test_metrics['classification_report']))    
    def stats(arr):
        arr_filtered = [x for x in arr if x is not None]
        return (np.mean(arr_filtered), np.max(arr_filtered), np.min(arr_filtered), np.std(arr_filtered)) if arr_filtered else (None, None, None, None)

    mean_acc, best_acc, worst_acc, std_acc = stats(accs)
    mean_bal, best_bal, worst_bal, std_bal = stats(bal_accs)
    mean_f1, best_f1, worst_f1, std_f1 = stats(f1s)
    mean_auc, best_auc, worst_auc, std_auc = stats(aucs)

    # Pour matrice de confusion et classification report, stocker le premier fold (exemple)
    conf_matrix_str = confusion_matrices[0] if confusion_matrices else ""
    class_report_str = classification_reports[0] if classification_reports else ""

    # Enregistrement final dans CSV
    final_metrics = {
        'Dataset': name,
        'Mean_Accuracy': f"{mean_acc:.4f}" if mean_acc is not None else "",
        'Best_Accuracy': f"{best_acc:.4f}" if best_acc is not None else "",
        'Worst_Accuracy': f"{worst_acc:.4f}" if worst_acc is not None else "",
        'Std_Accuracy': f"{std_acc:.4f}" if std_acc is not None else "",
        'Mean_Balanced_Accuracy': f"{mean_bal:.4f}" if mean_bal is not None else "",
        'Best_Balanced_Accuracy': f"{best_bal:.4f}" if best_bal is not None else "",
        'Worst_Balanced_Accuracy': f"{worst_bal:.4f}" if worst_bal is not None else "",
        'Std_Balanced_Accuracy': f"{std_bal:.4f}" if std_bal is not None else "",
        'Mean_F1_Score': f"{mean_f1:.4f}" if mean_f1 is not None else "",
        'Best_F1_Score': f"{best_f1:.4f}" if best_f1 is not None else "",
        'Worst_F1_Score': f"{worst_f1:.4f}" if worst_f1 is not None else "",
        'Std_F1_Score': f"{std_f1:.4f}" if std_f1 is not None else "",
        'Mean_AUC': f"{mean_auc:.4f}" if mean_auc is not None else "",
        'Best_AUC': f"{best_auc:.4f}" if best_auc is not None else "",
        'Worst_AUC': f"{worst_auc:.4f}" if worst_auc is not None else "",
        'Std_AUC': f"{std_auc:.4f}" if std_auc is not None else "",
        'Confusion_Matrix': conf_matrix_str,
        'Classification_Report': class_report_str
    }
    csv_logger.log_metrics(final_metrics)
    with open(csv_filename, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerow({
            'Dataset': name,
            **final_metrics
        })
    csv_logger.save()
    torch.save(best_model.state_dict(), f"models/{name}.pkl")
    print(f"Tous les résultats ont été enregistrés dans {csv_filename}.")



