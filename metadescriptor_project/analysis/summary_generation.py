import os
import pandas as pd
import numpy as np
from src.utils.colors import RED, COLOR_OFF


class SummaryConfig:
    """Configuration class for summary generation parameters."""
    
    def __init__(self):
        self.min_sups = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        self.types = ["NORMAL", "GSPAN", "CGSPAN"]
        self.nivs = [str(min_sup) for min_sup in self.min_sups]
        self.categories = ["num_nodes", "num_edges", "num_label_differents",  "density", "diameter", 'wiener_index', 'clustering_coef', 'traingle', 'kcore_max', 'mean_degree_centrality', 'mean_betweenness', 'mean_pagerank', 'mean_closeness', 'fiedler_value', 'spectral_radius', 'trace_laplacien', 'nb_zero_eigenvalues', "label"]
        self.node_label =[
        'Sb', 'Au', 'Eu', 'Sr', 'Tb', 'Nb', 'O+', 'O-', 'Cs', 'H', 'Rb',
        'Pb', 'S', 'N-', 'N+', 'N', 'Tl', 'C', 'Mg', 'Fe', 'K', 'PH1+', 'O',
        'La', 'Ru', 'Pt', 'B', 'Gd', 'Te', 'CH1', 'PH1', 'Li', 'Ni', 'Am',
        'As', 'Hg', 'Cr', 'U', 'S+', 'Ac', 'Mn', 'Ge', 'Pd', 'OH1+', 'W',
        'Ga', 'P+', 'Si', 'NH1+', 'Bi', 'Ce', 'Ir', 'Cd', 'Cl', 'I', 'Ta',
        'Na', 'Re', 'Cu', 'Ti', 'Sn', 'Ag', 'In', 'NH2+', 'Al', 'Y', 'Nd',
        'Br', 'Po', 'Co', 'NH3+', 'NH1-', 'S-', 'Se', 'SH1+', 'V', 'PH2', 'Zn',
        'Ba', 'Th', 'Rh', 'F', 'CH2-', 'Sc', 'Hf', 'CH1-', 'Pr', 'Mo', 'Fr',
        'Zr', 'Ho', 'Dy', 'Yb', 'Er', 'Sm', 'P', 'Os', 'Be', 'Ca'
        ]
        #self.categories += self.node_label
        self.moments = ["min", "mean", "std", "skewness", "kurtosis", "max"]
        self.other_categories = ["Count",  'dataset_nb_classes',
                    'dataset_prop_classe_majorite',
                    'dataset_label_entropy',
                ]


class FileProcessor:
    """Class for processing and organizing CSV files."""
    
    @staticmethod
    def get_csv_files():
        """Get and organize CSV files from SPMF directory."""
        files_csv = os.listdir("SPMF")
        files_csv = sorted([file for file in files_csv if file.endswith(".csv") and "NORMAL" not in file])
        
        files_csv_normal = os.listdir("SPMF")
        files_csv_normal = sorted([file for file in files_csv_normal if file.endswith(".csv") and "NORMAL" in file])
        
        combined = [(files_csv[i], files_csv[i + 1]) for i in range(0, len(files_csv), 2)]
        
        return files_csv, files_csv_normal, combined
    
    @staticmethod
    def organize_files_by_dataset(good_datasets_sorted, combined, files_csv_normal, min_sups):
        """Organize files by dataset."""
        combined_by_dataset = {}
        starting = 0
        
        for dataset in good_datasets_sorted:
            files = combined[starting : starting + len(min_sups) * 2]
            starting += len(min_sups) * 2
            
            # Add normal files for this dataset
            normal_files = [
                (files_csv_normal[i], files_csv_normal[i + 1]) 
                for i in range(0, len(files_csv_normal), 2) 
                if files_csv_normal[i].startswith(dataset + "_NORMAL")
            ]
            
            combined_by_dataset[dataset] = files + normal_files
        
        print(f"Total datasets processed: {len(combined_by_dataset)}")
        print(combined_by_dataset)
        
        return combined_by_dataset


class ColumnGenerator:
    """Class for generating column names for the summary DataFrame."""
    
    def __init__(self, config):
        self.config = config
    
    def generate_columns(self):
        """Generate all column names for the summary DataFrame."""
        columns = ["Dataset"]
        
        for type_ in self.config.types:
            if type_ == "NORMAL":
                columns.extend(self._generate_normal_columns(type_))
            else:
                columns.extend(self._generate_algorithm_columns(type_))
        
        return columns
    
    def _generate_normal_columns(self, type_):
        """Generate columns for NORMAL type."""
        columns = []
        
        # Regular categories
        for cat in self.config.categories:
            if cat == "label":
                continue
            elif cat == "num_label_differents":
                for moment in self.config.moments:
                    columns.append(f"{type_}_num_etiquette_sommets_differents_{moment}")
            else:
                for moment in self.config.moments:
                    columns.append(f"{type_}_{cat}_{moment}")
        
        # Other categories
        for other in self.config.other_categories:
            columns.append(f"{type_}_{other}")
        
        return columns
    
    def _generate_algorithm_columns(self, type_):
        """Generate columns for GSPAN/CGSPAN types."""
        columns = []
        
        for niv in self.config.nivs:
            # Regular categories
            for cat in self.config.categories:
                if cat == "label":
                    for moment in self.config.moments:
                        columns.append(f"{type_}_{niv}_frequence_{moment}")
                elif cat == "num_label_differents":
                    for moment in self.config.moments:
                        columns.append(f"{type_}_{niv}_num_etiquette_sommets_differents_{moment}")
                else:
                    for moment in self.config.moments:
                        columns.append(f"{type_}_{niv}_{cat}_{moment}")
            
            # Other categories
            for other in self.config.other_categories:
                columns.append(f"{type_}_{niv}_{other}")
        
        return columns


class DataProcessor:
    """Class for processing data and creating summary rows."""
    
    def __init__(self, config, columns):
        self.config = config
        self.columns = columns
    
    def process_datasets(self, combined_by_dataset):
        """Process all datasets and create summary DataFrame."""
        pd_data = pd.DataFrame(columns=self.columns)
        
        print("\n\n----- Processing datasets -----\n\n")
        
        for dataset, files in combined_by_dataset.items():
            print(f"Current dataset: '{dataset}'")
            row = self._process_single_dataset(dataset, files)
            pd_data = pd.concat([pd_data, pd.DataFrame([row])], ignore_index=True)
        
        print("All datasets processed. Final shape of pd_data:", pd_data.shape)
        return pd_data
    
    def _process_single_dataset(self, dataset, files):
        """Process a single dataset and return its row data."""
        row = {"Dataset": dataset}
        
        for file_infos, file_stats in files:
            file_data = self._load_file_data(file_infos, file_stats)
            if file_data is None:
                continue
            
            infos, stats, type_ = file_data
            
            if type_ == "NORMAL":
                self._process_normal_file(row, infos, stats, type_)
            else:
                niv = file_stats.split("_")[-2]
                self._process_algorithm_file(row, infos, stats, type_, niv)
        
        return row
    
    def _load_file_data(self, file_infos, file_stats):
        """Load and identify file data."""
        try:
            infos = pd.read_csv(f"SPMF/{file_infos}", index_col=0)
            stats = pd.read_csv(f"SPMF/{file_stats}", index_col=0)
        except Exception as e:
            print(f"{RED}Error loading files {file_infos}, {file_stats}: {e}{COLOR_OFF}")
            return None
        
        # Determine type
        if "NORMAL" in file_infos:
            type_ = "NORMAL"
        elif "CGSPAN" in file_stats:
            type_ = "CGSPAN"
        elif "GSPAN" in file_stats:
            type_ = "GSPAN"
        else:
            print(f"Unknown type_ in file: {file_stats}")
            return None
        
        return infos, stats, type_
    
    def _process_normal_file(self, row, infos, stats, type_):
        """Process NORMAL type files."""
        # Process categories
        for cat in self.config.categories:
            if cat == "label":
                continue
            
            for moment in self.config.moments:
                column_name = self._get_normal_column_name(type_, cat, moment)
                value = self._get_stat_value(stats, moment, cat, column_name)
                row[column_name] = value
        
        # Process other categories
        for other in self.config.other_categories:
            column_name = f"{type_}_{other}"
            if column_name not in self.columns:
                print(f"Column {column_name} not in COLUMNS, skipping")
                continue
            
            if other == "Count":
                row[column_name] = infos.shape[0]
            else:
                row[column_name] = infos[other][0]
    
    def _process_algorithm_file(self, row, infos, stats, type_, niv):
        """Process GSPAN/CGSPAN type files."""
        # Process categories
        for cat in self.config.categories:
            for moment in self.config.moments:
                column_name = self._get_algorithm_column_name(type_, niv, cat, moment)
                value = self._get_stat_value(stats, moment, cat, column_name)
                row[column_name] = value
        
        # Process other categories
        for other in self.config.other_categories:
            column_name = f"{type_}_{niv}_{other}"
            if column_name not in self.columns:
                print(f"{RED}Column {column_name} not in COLUMNS, skipping{COLOR_OFF}")
                continue
            
            if other == "Count":
                row[column_name] = infos.shape[0]
            else:
                row[column_name] = 0
    
    def _get_normal_column_name(self, type_, cat, moment):
        """Get column name for NORMAL type."""
        if cat == "num_label_differents":
            return f"{type_}_num_etiquette_sommets_differents_{moment}"
        else:
            return f"{type_}_{cat}_{moment}"
    
    def _get_algorithm_column_name(self, type_, niv, cat, moment):
        """Get column name for GSPAN/CGSPAN type."""
        if cat == "label":
            return f"{type_}_{niv}_frequence_{moment}"
        elif cat == "num_label_differents":
            return f"{type_}_{niv}_num_etiquette_sommets_differents_{moment}"
        else:
            return f"{type_}_{niv}_{cat}_{moment}"
    
    def _get_stat_value(self, stats, moment, cat, column_name):
        """Get statistical value from stats DataFrame."""
        if column_name not in self.columns:
            print(f"{RED}Column {column_name} not in COLUMNS, skipping{COLOR_OFF}")
            return 0
        
        if moment in stats.index and cat in stats.columns:
            return float(stats.loc[moment, cat])
        else:
            print(f"{RED}Column {column_name} not found in stats, setting to 0{COLOR_OFF}")
            return 0


class SummaryGenerator:
    """Main class for generating summaries."""
    
    def __init__(self):
        self.config = SummaryConfig()
        self.column_generator = ColumnGenerator(self.config)
        self.columns = self.column_generator.generate_columns()
        self.data_processor = DataProcessor(self.config, self.columns)
    
    def generate(self, good_datasets_sorted=None):
        """Generate summary from datasets."""
        if good_datasets_sorted is None:
            print("Warning: good_datasets_sorted is None")
            return
        
        # Get and organize files
        files_csv, files_csv_normal, combined = FileProcessor.get_csv_files()
        combined_by_dataset = FileProcessor.organize_files_by_dataset(
            good_datasets_sorted, combined, files_csv_normal, self.config.min_sups
        )
        
        # Process datasets
        pd_data = self.data_processor.process_datasets(combined_by_dataset)
        
        # Clean and save data
        self._clean_and_save_data(pd_data)
        
        return pd_data
    
    def _clean_and_save_data(self, pd_data):
        """Clean data and save to CSV."""
        pd_data.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
        pd_data.to_csv("summary.csv", index=False)
        print("Summary saved to summary.csv")


# Public interface function (maintains compatibility)
def generate_summary(GOOD_DATASETS_SORTED=None):
    """
    Generate summary from datasets.
    
    Args:
        GOOD_DATASETS_SORTED: List of dataset names to process
        
    Returns:
        DataFrame: Processed summary data
    """
    generator = SummaryGenerator()
    return generator.generate(GOOD_DATASETS_SORTED)


# Alternative interface for more control
class SummaryBuilder:
    """Builder class for more flexible summary generation."""
    
    def __init__(self):
        self.config = SummaryConfig()
        self.reset()
    
    def reset(self):
        """Reset builder to initial state."""
        self.column_generator = ColumnGenerator(self.config)
        self.columns = self.column_generator.generate_columns()
        self.data_processor = DataProcessor(self.config, self.columns)
        return self
    
    def with_min_sups(self, min_sups):
        """Set custom minimum support values."""
        self.config.min_sups = min_sups
        self.config.nivs = [str(min_sup) for min_sup in min_sups]
        return self.reset()
    
    def with_types(self, types):
        """Set custom algorithm types."""
        self.config.types = types
        return self.reset()
    
    def with_categories(self, categories):
        """Set custom categories."""
        self.config.categories = categories
        return self.reset()
    
    def with_moments(self, moments):
        """Set custom statistical moments."""
        self.config.moments = moments
        return self.reset()
    
    def build(self, good_datasets_sorted, output_file="summary.csv"):
        """Build summary with current configuration."""
        generator = SummaryGenerator()
        generator.config = self.config
        generator.column_generator = self.column_generator
        generator.columns = self.columns
        generator.data_processor = self.data_processor
        
        pd_data = generator.generate(good_datasets_sorted)
        
        if output_file != "summary.csv":
            pd_data.to_csv(output_file, index=False)
            print(f"Summary saved to {output_file}")
        
        return pd_data
