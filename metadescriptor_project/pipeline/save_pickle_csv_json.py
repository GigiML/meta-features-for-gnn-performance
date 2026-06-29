import os
import pickle
import json

def save_all_pickles_csv_json():
    pickles_dir = "SPMF"
    all_pickles = {}
    for file in os.listdir(pickles_dir):
        if file.endswith(".pkl"):
            with open(os.path.join(pickles_dir, file), "rb") as f:
                all_data = pickle.load(f)
                all_pickles[file] = all_data
                print(f"Loaded {file} with {len(all_data)} entries")

                for key, value in all_data.items():
                    if hasattr(value, 'to_csv'):
                        value.to_csv(f"{key}.csv", index=True)

    for name, data in all_pickles.items():
        for key, value in data.items():
            if hasattr(value, 'to_json'):
                data[key] = value.to_json(orient="records", lines=True)

    for name, data in all_pickles.items():
        with open("SPMF/" + name + ".json", "w") as f:
            json.dump(data, f)