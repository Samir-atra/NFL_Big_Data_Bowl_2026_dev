"""
Script to patch the `predictor.ipynb` Jupyter notebook for sequence data padding.

This script directly modifies the notebook's JSON structure to update the 
`preprocess_features` function within a specific code cell. It replaces the original
sequence extraction logic with new code that correctly handles historical data
slicing and prepends zero-padding when the full history (`SEQUENCE_LENGTH`) is not
available at the beginning of a play, ensuring the input sequence array has a 
consistent length.
"""
import json
import os

# NOTE: Since this script injects code that uses numpy functions (np.zeros, np.vstack)
# those functions must be available within the notebook execution environment.

# --- Configuration ---
# Path to the notebook file to be patched
file_path = '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/src/predictor.ipynb'

# --- Main Patching Logic ---

# 1. Load the notebook's JSON content
print(f"Loading notebook: {file_path}")
with open(file_path, 'r') as f:
    nb = json.load(f)

# 2. Find the target code cell containing the 'def preprocess_features' function
prep_cell = None
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        # Search for the function definition line
        if any('def preprocess_features' in line for line in cell['source']):
            prep_cell = cell
            break

if prep_cell:
    source = prep_cell['source']
    
    # 3. Find the starting line index of the original loop over prediction targets
    loop_start_idx = -1
    for i, line in enumerate(source):
        # Identify the start of the original loop
        if "for _, row_to_predict in test_df.iterrows():" in line:
            loop_start_idx = i
            break
    
    if loop_start_idx != -1:
        # Keep everything before the loop (function definition, setup, etc.)
        new_source = source[:loop_start_idx]
        
        # 4. Define and insert the new loop logic with padding implementation
        new_source.append("    for _, row_to_predict in test_df.iterrows():\n")
        new_source.append("        player_id = row_to_predict['nfl_id']\n")
        new_source.append("        frame_id = row_to_predict['frame_id']\n")
        new_source.append("        \n")
        new_source.append("        # Find the player's data and the exact frame we need to predict\n")
        new_source.append("        player_data_with_ids = processed_df_with_ids[processed_df_with_ids['nfl_id'] == player_id]\n")
        new_source.append("        prediction_frame_index = player_data_with_ids[player_data_with_ids['frame_id'] == frame_id].index[0]\n")
        new_source.append("        \n")
        new_source.append("        # The sequence consists of the `SEQUENCE_LENGTH` frames *before* the prediction frame\n")
        new_source.append("        start_idx = prediction_frame_index - SEQUENCE_LENGTH\n")
        new_source.append("        end_idx = prediction_frame_index\n")
        new_source.append("        \n")
        new_source.append("        if start_idx < 0:\n")
        new_source.append("            # If we don't have enough history, slice from the start\n")
        new_source.append("            sequence = processed_features_df.iloc[0:end_idx].values\n")
        new_source.append("            # Calculate necessary zero padding\n")
        new_source.append("            pad_width = SEQUENCE_LENGTH - len(sequence)\n")
        new_source.append("            if pad_width > 0:\n")
        new_source.append("                # Prepend zero padding to achieve the required SEQUENCE_LENGTH\n")
        new_source.append("                padding = np.zeros((pad_width, sequence.shape[1]))\n")
        new_source.append("                sequence = np.vstack([padding, sequence])\n")
        new_source.append("        else:\n")
        new_source.append("            # Slice the full sequence history from the purely numerical dataframe\n")
        new_source.append("            sequence = processed_features_df.iloc[start_idx:end_idx].values\n")
        new_source.append("        \n")
        new_source.append("        sequences.append(sequence)\n")
        new_source.append("\n")
        # NOTE: The original code returned np.array(sequences). Adding the import check if necessary.
        new_source.append("    return np.array(sequences)\n")
        
        prep_cell['source'] = new_source
        print("preprocess_features modified.")
    else:
        print("Loop start not found in preprocess_features.")
else:
    print("preprocess_features cell not found!")

# 5. Save the modified notebook JSON
print("Saving modified notebook...")
with open(file_path, 'w') as f:
    # Use indent=1 for readability and consistency with typical notebook format
    json.dump(nb, f, indent=1)

print("Notebook patched successfully.")
