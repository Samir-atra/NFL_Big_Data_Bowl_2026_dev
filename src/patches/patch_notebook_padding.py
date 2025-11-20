import json
import os

file_path = '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/src/predictor.ipynb'

with open(file_path, 'r') as f:
    nb = json.load(f)

# Modify preprocess_features (Cell 8)
# Search for cell containing "def preprocess_features"
prep_cell = None
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        if any('def preprocess_features' in line for line in cell['source']):
            prep_cell = cell
            break

if prep_cell:
    source = prep_cell['source']
    new_source = []
    # We will replace the loop part
    
    # Find the start of the loop
    loop_start_idx = -1
    for i, line in enumerate(source):
        if "for _, row_to_predict in test_df.iterrows():" in line:
            loop_start_idx = i
            break
    
    if loop_start_idx != -1:
        # Keep everything before the loop
        new_source = source[:loop_start_idx]
        
        # Add the new loop logic with padding
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
        new_source.append("            # If we don't have enough history, pad with the first frame or zeros\n")
        new_source.append("            # Here we slice from 0 to end_idx\n")
        new_source.append("            sequence = processed_features_df.iloc[0:end_idx].values\n")
        new_source.append("            # Pad with zeros at the beginning\n")
        new_source.append("            pad_width = SEQUENCE_LENGTH - len(sequence)\n")
        new_source.append("            if pad_width > 0:\n")
        new_source.append("                # Pad with the first available frame (repetition) or zeros. \n")
        new_source.append("                # Using zeros is safer if we assume missing history means 'nothing happened'\n")
        new_source.append("                # But repetition might be better for continuity. Let's use zero padding for now as it's standard.\n")
        new_source.append("                padding = np.zeros((pad_width, sequence.shape[1]))\n")
        new_source.append("                sequence = np.vstack([padding, sequence])\n")
        new_source.append("        else:\n")
        new_source.append("            # Slice the sequence from the purely numerical dataframe\n")
        new_source.append("            sequence = processed_features_df.iloc[start_idx:end_idx].values\n")
        new_source.append("        \n")
        new_source.append("        sequences.append(sequence)\n")
        new_source.append("\n")
        new_source.append("    return np.array(sequences)")
        
        prep_cell['source'] = new_source
        print("preprocess_features modified.")
    else:
        print("Loop start not found in preprocess_features.")
else:
    print("preprocess_features cell not found!")

with open(file_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook patched successfully.")
