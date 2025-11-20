import json
import os

file_path = '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/src/predictor.ipynb'

with open(file_path, 'r') as f:
    nb = json.load(f)

# Modify Imports (Cell 1)
# We assume Cell 1 is the code cell with imports based on inspection
# But let's search for the cell containing "import tensorflow" just to be safe
import_cell = None
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        if any('import tensorflow' in line for line in cell['source']):
            import_cell = cell
            break

if import_cell:
    source = import_cell['source']
    # Check if already imported
    if not any('SimpleImputer' in line for line in source):
        # Find where to insert. After 'from sklearn.pipeline ...'
        inserted = False
        for i, line in enumerate(source):
            if 'from sklearn.pipeline import Pipeline' in line:
                source.insert(i + 1, "from sklearn.impute import SimpleImputer\n")
                inserted = True
                break
        if not inserted:
            # If not found, append to the end of imports
            source.append("from sklearn.impute import SimpleImputer\n")
    print("Imports modified.")
else:
    print("Import cell not found!")

# Modify create_preprocessor (Cell 3)
# Search for cell containing "def create_preprocessor"
prep_cell = None
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        if any('def create_preprocessor' in line for line in cell['source']):
            prep_cell = cell
            break

if prep_cell:
    source = prep_cell['source']
    new_source = []
    modified = False
    for line in source:
        if 'numerical_transformer = StandardScaler()' in line:
            new_source.append("    numerical_transformer = Pipeline(steps=[\n")
            new_source.append("        ('imputer', SimpleImputer(strategy='mean')),\n")
            new_source.append("        ('scaler', StandardScaler())\n")
            new_source.append("    ])\n")
            modified = True
        else:
            new_source.append(line)
    
    prep_cell['source'] = new_source
    if modified:
        print("create_preprocessor modified.")
    else:
        print("Target line in create_preprocessor not found (or already modified).")
else:
    print("create_preprocessor cell not found!")

with open(file_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook patched successfully.")
