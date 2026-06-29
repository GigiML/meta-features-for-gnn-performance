def download_zip(dataset):
    url = f"https://www.chrsmrrs.com/graphkerneldatasets/{dataset}.zip"
    # Nom du fichier ZIP téléchargé
    zip_file = f"{dataset}.zip"
    # Dossier où extraire les fichiers
    extract_dir = "data/zip"
    
    # 1. Téléchargement du fichier ZIP
    print(f"Téléchargement de {url}...")
    response = requests.get(url)
    if response.status_code == 200:
        with open(zip_file, 'wb') as f:
            f.write(response.content)
        print(f"Fichier ZIP téléchargé : {zip_file}")
    else:
        print(f"Erreur lors du téléchargement : {response.status_code}")
        exit()
    
    # 2. Extraction du ZIP
    print(f"Extraction de {zip_file} dans {extract_dir}...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Fichiers extraits dans {extract_dir}")
    
    # 3. (Optionnel) Suppression du ZIP après extraction
    if os.path.exists(zip_file):
        os.remove(zip_file)
        print(f"Fichier ZIP supprimé : {zip_file}")
#Verifier que chaque dossier à un fichier readme
for dataset in datasets:
    l = !ls data/zip/{dataset} | grep -E 'readme.*|README.*\.txt$'
    if l ==[]:
    print(dataset)
#Extraire le mapping qui nous intéresse + le mettre en dict


import re

def extract_section(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Première tentative : extraire texte entre "node label" + "Component 0 :" et avant le prochain label
        pattern = re.compile(
            r"(?:node label).*?Component\s*0\s*:\s*(.*?)(?=\n\s*\w+ labels|node\Z)",
            re.DOTALL | re.IGNORECASE
        )
        matches = pattern.findall(text)
        if matches:
            return matches[0].strip()
        
        # Deuxième tentative : extraire texte après "Component 0 :" avant "==="
        pattern = re.compile(
            r"(?=.*node label.*Component\s*0\s*:).*?Component\s*0\s*:\s*(.*?)(?=\n\s*===)",
            re.DOTALL | re.IGNORECASE
        )
        match = pattern.findall(text)
        if match:
            return match[0].strip()
        
        # Troisième tentative : extraire texte entre "Node labels:" et "Edge labels:" ou fin de fichier
        pattern = re.compile(
            r"Node labels:(.*?)(?=\nEdge labels:|\Z)",
            re.DOTALL | re.IGNORECASE
        )
        matches = pattern.findall(text)
        if matches:
            return matches[0].strip()
        
        return None

    except Exception as e:
        print(f"Erreur avec {filename}: {e}")
        return None


def list_to_dict(l):
    if l[0] == 2 :
        return {ele[0]: ele[1] for ele in l}
    else:
        {part[0]: part[1] for ele in l if (part := ele[0].split(' ', 1)) and len(part) == 2}


datasets = os.listdir("data/TUDataset")
metadata = {}
for dataset in datasets:
    print(dataset)
    readme_files = [
        f for f in os.listdir(f"data/zip/{dataset}")
        if re.search(r'readme.*|README.*\.txt$', f, re.I)
    ]
    if readme_files:
        filename = f"data/zip/{dataset}/{readme_files[0]}"
        print(f"Fichier README trouvé : {filename}")
        map_text = extract_section(filename)
        if map_text:
            result = list(map(lambda x: x.lstrip().split('\t'), map_text.split('\n')))
            metadata[dataset] = list_to_dict(result)
            if  len(metadata[dataset])==0:
                print(result)
        else:
            print("Aucune correspondance extraite.")
    else:
        print("Aucun fichier README trouvé.")

inverse = list()
for key in data.keys():
    for k, v in data[key].items():
        if k.isdigit():
            print(k, key)
        elif v.isdigit():
            print(v, key)
            inverse.append(key)
        else:
            print(key)
        break;



for key in inverse :
    su_dict[key] = {v:k for k,v in su_dict[key].items()}

for key in su_dict.keys() :
    su_dict[key] = {int(v):k.replace(" ","") for k,v in su_dict[key].items()}

atomes = set()

for k,v  in data.items():
    atomes.update(data[k].values())
 

with open("data/zip/atomes.pkl", "wb") as f:
    pickle.dump(atomes, f)

