import json
import os
import pandas as pd
import numpy as np
import random

def load_my_columns_from_file(filepath="tmp/MY_COLUMNS_complete.json"):
    """
    Charge MY_COLUMNS depuis un fichier JSON.
    
    Args:
        filepath (str): Chemin vers le fichier JSON contenant MY_COLUMNS
        
    Returns:
        dict: Dictionnaire MY_COLUMNS ou None si le fichier n'existe pas
    """
    try:
        if os.path.exists(filepath):
            print(f"Chargement de MY_COLUMNS depuis {filepath}")
            with open(filepath, 'r') as f:
                my_columns = json.load(f)
            print(f"MY_COLUMNS chargé avec {len(my_columns)} seuils")
            return my_columns
        else:
            print(f"Fichier {filepath} non trouvé. Utilisation des valeurs par défaut.")
            return None
    except Exception as e:
        print(f"Erreur lors du chargement de {filepath}: {e}")
        return None

def save_my_columns_to_file(my_columns, filepath="tmp/MY_COLUMNS_complete.json"):
    """
    Sauvegarde MY_COLUMNS dans un fichier JSON.
    
    Args:
        my_columns (dict): Dictionnaire MY_COLUMNS à sauvegarder
        filepath (str): Chemin où sauvegarder le fichier
    """
    try:
        # Créer le dossier si nécessaire
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(my_columns, f, indent=2)
        
        print(f"MY_COLUMNS sauvegardé dans {filepath}")
        return filepath
    except Exception as e:
        print(f"Erreur lors de la sauvegarde dans {filepath}: {e}")
        return None

def save_selected_descriptors(selected_descriptors, threshold, method="clustering", filepath=None):
    """
    Sauvegarde les descripteurs sélectionnés dans un fichier JSON.
    
    Args:
        selected_descriptors: Liste de descripteurs sélectionnés
        threshold (float): Seuil utilisé pour la sélection
        method (str): Méthode utilisée pour la sélection
        filepath (str): Chemin de sauvegarde (optionnel)
        
    Returns:
        str: Chemin du fichier sauvegardé
    """
    if filepath is None:
        filepath = f"tmp/selected_descriptors_{method}_{threshold:.2f}.json"
    
    # Créer le dossier si nécessaire
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Préparer les données à sauvegarder
    if isinstance(selected_descriptors, list):
        if selected_descriptors and isinstance(selected_descriptors[0], tuple):
            data = {
                "method": method,
                "threshold": threshold,
                "descriptors": [desc[1] for desc in selected_descriptors],
                "full_info": selected_descriptors
            }
        else:
            data = {
                "method": method,
                "threshold": threshold,
                "descriptors": selected_descriptors
            }
    else:
        data = {
            "method": method,
            "threshold": threshold,
            "descriptors": selected_descriptors
        }
    
    # Sauvegarder en JSON
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Descripteurs sauvegardés dans: {filepath}")
        return filepath
    except Exception as e:
        print(f"Erreur lors de la sauvegarde des descripteurs: {e}")
        return None

def set_random_seeds(seed=None):
    """
    Configure les seeds aléatoires pour assurer la reproductibilité.
    
    Args:
        seed (int): Seed spécifique, ou None pour une seed aléatoire
        
    Returns:
        int: La seed utilisée
    """
    if seed is None:
        seed = random.randint(1, 10000)
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Si pandas est disponible
    try:
        import pandas as pd
        # Pas de seed directe pour pandas, mais on peut l'utiliser pour les échantillonnages
    except ImportError:
        pass
    
    print(f"Seeds configurées: {seed}")
    return seed

def run_multiple_experiments(experiment_func, n_runs=5, **kwargs):
    """
    Lance une expérience plusieurs fois avec des seeds différentes.
    
    Args:
        experiment_func: Fonction d'expérience à appeler
        n_runs (int): Nombre d'exécutions
        **kwargs: Arguments à passer à experiment_func
        
    Returns:
        list: Liste des résultats de chaque exécution
    """
    results = []
    seeds_used = []
    
    print(f"=== LANCEMENT DE {n_runs} EXPÉRIENCES ===")
    
    for i in range(n_runs):
        print(f"\n--- Expérience {i+1}/{n_runs} ---")
        
        # Configurer une seed aléatoire différente pour chaque run
        seed = set_random_seeds()
        seeds_used.append(seed)
        
        # Lancer l'expérience
        try:
            result = experiment_func(**kwargs)
            results.append({
                'run': i+1,
                'seed': seed,
                'result': result
            })
            print(f"✓ Expérience {i+1} terminée avec seed {seed}")
            
        except Exception as e:
            print(f"✗ Erreur dans l'expérience {i+1}: {e}")
            results.append({
                'run': i+1,
                'seed': seed,
                'result': None,
                'error': str(e)
            })
    
    # Calculer les statistiques
    valid_results = [r['result'] for r in results if r['result'] is not None]
    
    print(f"\n=== RÉSUMÉ DES {n_runs} EXPÉRIENCES ===")
    print(f"Expériences réussies: {len(valid_results)}/{n_runs}")
    print(f"Seeds utilisées: {seeds_used}")
    
    if valid_results:
        if isinstance(valid_results[0], (int, float)):
            # Résultats numériques - calculer moyenne et std
            mean_result = np.mean(valid_results)
            std_result = np.std(valid_results)
            print(f"Résultat moyen: {mean_result:.4f} ± {std_result:.4f}")
        else:
            # Résultats complexes
            print(f"Résultats obtenus pour analyse détaillée")
    
    return results


def rename_and_add_columns(df, rename_dict, columns_set):
    """
    Renames columns in the DataFrame according to rename_dict,
    then adds missing columns from columns_set initialized with NaN.
    
    Args:
        df (pd.DataFrame): The DataFrame to modify
        rename_dict (dict): Dictionary {old_name: new_name} for renaming columns
        columns_set (set): Set of columns to ensure present in the DataFrame
    
    Returns:
        pd.DataFrame: Modified DataFrame
    """
    # Rename columns
    df = df.rename(columns=rename_dict)
    
    # Add missing columns with NaN values
    for col in columns_set:
        if col not in df.columns:
            df[col] = np.nan
    
    return df