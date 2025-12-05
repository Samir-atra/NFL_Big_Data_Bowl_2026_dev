"""
Script to apply necessary patches to the 'src/predictor.ipynb' notebook.

This patch addresses critical preprocessing issues by performing two main actions:
1. Ensures the `SimpleImputer` is imported from `sklearn.impute`.
2. Modifies the `create_preprocessor` function to include a `SimpleImputer` with 
   a 'mean' strategy within the `numerical_transformer` pipeline, handling missing 
   numerical values before scaling.
"""
import json
import os

# --- Configuration ---
# Path to the notebook file to be patched
file_path = '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/src/predictor.ipynb'

# --- Main Patching Logic ---

# 1. Load the notebook's JSON content
print(f"Loading notebook: {file_path}")
with open(file_path, 'r') as f:
    nb = json.load(f)

# 2. Modify Imports (Cell 1)
# Search for the cell containing "import tensorflow" or other key imports
import_cell = None
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        if any('import tensorflow' in line for line in cell['source']):
            import_cell = cell
            break

if import_cell:
    source = import_cell['source']
    # Check if SimpleImputer is already imported
    if not any('SimpleImputer' in line for line in source):
        # Find where to insert the new import (ideally after other sklearn imports)
        inserted = False
        for i, line in enumerate(source):
            if 'from sklearn.pipeline import Pipeline' in line:
                source.insert(i + 1, "from sklearn.impute import SimpleImputer\n")
                inserted = True
                break
        if not inserted:
            # If the target line is not found, append the import
            source.append("from sklearn.impute import SimpleImputer\n")
    print("Imports modified: SimpleImputer ensured.")
else:
    print("Import cell not found!")

# 3. Modify create_preprocessor function (Cell ~3)
# Search for the cell containing "def create_preprocessor"
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
    # Look for the definition of the numerical transformer
    for line in source:
        if 'numerical_transformer = StandardScaler()' in line:
            # Replace the simple StandardScaler with a Pipeline including SimpleImputer
            new_source.append("    numerical_transformer = Pipeline(steps=[\n")
            new_source.append("        ('imputer', SimpleImputer(strategy='mean')),\n")
            new_source.append("        ('scaler', StandardScaler())\n")
            new_source.append("    ])\n")
            modified = True
        else:
            new_source.append(line)
    
    prep_cell['source'] = new_source
    if modified:
        print("create_preprocessor modified: Imputer added to numerical pipeline.")
    else:
        print("Target line in create_preprocessor not found (or already modified).")
else:
    print("create_preprocessor cell not found!")

# 4. Save the modified notebook JSON
print("Saving modified notebook...")
with open(file_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook patched successfully.")
