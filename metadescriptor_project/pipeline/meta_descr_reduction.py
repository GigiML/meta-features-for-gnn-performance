import pandas as pd

def reduction_dimension(
        correlation_file="src/tmp/correlation_matrix_column_by_column.csv", 
        importance_file="src/tmp/importance_variables.csv",
        threshold=0.9):
    
    correlation_df = pd.read_csv(correlation_file, index_col=0)
    importance_df = pd.read_csv(importance_file)
    importance_df.rename(columns={"Unnamed: 0": "feature"}, inplace=True)
    importance_sorted = importance_df.sort_values(by="Mean").reset_index(drop=True)
    features_ordered = importance_sorted["feature"].tolist()

    selected_features = []
    excluded_features = set()

    for feature in features_ordered:
        if feature in correlation_df.columns and feature not in excluded_features:
            # Vérifier si la variable a AU MOINS UNE corrélation > threshold avec une AUTRE variable
            high_corr = correlation_df.loc[feature].abs() > threshold
            high_corr[feature] = False  # Exclure l'auto-corrélation (toujours 1)
            
            # if not high_corr.any():
            #     # Si la variable n'est pas corrélée à d'autres, on la retire (ne rien faire)
            #     continue

            # Sinon, on la sélectionne
            selected_features.append(feature)

            # Exclure les autres features fortement corrélées à celle-ci
            correlated_features = correlation_df.columns[high_corr].tolist()
            excluded_features.update(correlated_features)

    print(f"Selected {len(selected_features)} features with threshold {threshold}")
    return selected_features
