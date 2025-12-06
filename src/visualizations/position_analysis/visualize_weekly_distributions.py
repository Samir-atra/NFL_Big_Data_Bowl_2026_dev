import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# Ensure backend acts non-interactively
import matplotlib
matplotlib.use('Agg')

class PositionAnalyzer:
    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def convert_height(self, height_str):
        if pd.isna(height_str): return None
        try:
            if isinstance(height_str, str) and '-' in height_str:
                feet, inches = map(int, height_str.split('-'))
                return feet * 12 + inches
            return float(height_str)
        except:
            return None

    def analyze_week(self, file_path):
        week_name = os.path.basename(file_path).replace('input_', '').replace('.csv', '')
        print(f"Processing {week_name}...")
        
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
            return

        # Preprocessing
        df['player_height_inches'] = df['player_height'].apply(self.convert_height)
        
        # Determine output directory for this week
        week_out = os.path.join(self.output_dir, week_name)
        os.makedirs(week_out, exist_ok=True)
        
        # --- Static Features (per player) ---
        # Height and Weight should be unique per player, but let's take one entry per nfl_id
        df_player_static = df.drop_duplicates(subset=['nfl_id']).copy()

        # 1. Height vs Position
        plt.figure(figsize=(16, 8))
        sns.boxplot(x='player_position', y='player_height_inches', data=df_player_static)
        plt.title(f'Player Height by Position ({week_name})')
        plt.xlabel('Position')
        plt.ylabel('Height (inches)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(week_out, 'height_by_position.png'))
        plt.close()

        # 2. Weight vs Position
        plt.figure(figsize=(16, 8))
        sns.boxplot(x='player_position', y='player_weight', data=df_player_static)
        plt.title(f'Player Weight by Position ({week_name})')
        plt.xlabel('Position')
        plt.ylabel('Weight (lbs)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(week_out, 'weight_by_position.png'))
        plt.close()

        # --- Dynamic Features (per frame) ---
        # S, A, Dir, O, X, Y
        
        # 3. Speed vs Position
        plt.figure(figsize=(16, 8))
        sns.boxplot(x='player_position', y='s', data=df)
        plt.title(f'Speed by Position ({week_name})')
        plt.xlabel('Position')
        plt.ylabel('Speed (yards/s)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(week_out, 'speed_by_position.png'))
        plt.close()

        # 4. Acceleration vs Position
        plt.figure(figsize=(16, 8))
        sns.boxplot(x='player_position', y='a', data=df)
        plt.title(f'Acceleration by Position ({week_name})')
        plt.xlabel('Position')
        plt.ylabel('Acceleration (yards/s^2)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(week_out, 'acceleration_by_position.png'))
        plt.close()
        
        # 5. Orientation (o) vs Position
        if 'o' in df.columns:
            plt.figure(figsize=(16, 8))
            sns.boxplot(x='player_position', y='o', data=df)
            plt.title(f'Orientation (o) by Position ({week_name})')
            plt.xlabel('Position')
            plt.ylabel('Orientation (degrees)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(week_out, 'orientation_by_position.png'))
            plt.close()

        # 6. Direction (dir) vs Position
        if 'dir' in df.columns:
            plt.figure(figsize=(16, 8))
            sns.boxplot(x='player_position', y='dir', data=df)
            plt.title(f'Direction (dir) by Position ({week_name})')
            plt.xlabel('Position')
            plt.ylabel('Direction (degrees)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(week_out, 'direction_by_position.png'))
            plt.close()

        # --- Categorical/Play Features ---
        # Role and Side might vary by play, so drop duplicates by (nfl_id, play_id)
        df_play_static = df.drop_duplicates(subset=['play_id', 'nfl_id']).copy()
        
        # 7. Role Distribution
        if 'player_role' in df.columns:
            plt.figure(figsize=(16, 8))
            # Create a cross-tabulation
            ct = pd.crosstab(df_play_static['player_position'], df_play_static['player_role'])
            # Normalize to get percentages if desired, or just raw counts. Raw counts shows volume.
            ct.plot(kind='bar', stacked=True, figsize=(16, 8), cmap='viridis')
            plt.title(f'Player Role Distribution by Position ({week_name})')
            plt.xlabel('Position')
            plt.ylabel('Count of Plays')
            plt.xticks(rotation=45)
            plt.legend(title='Player Role', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(week_out, 'role_by_position.png'))
            plt.close()

        # 8. Side Distribution
        if 'player_side' in df.columns:
            plt.figure(figsize=(16, 8))
            ct = pd.crosstab(df_play_static['player_position'], df_play_static['player_side'])
            ct.plot(kind='bar', stacked=True, figsize=(16, 8), cmap='Set2')
            plt.title(f'Player Side Distribution by Position ({week_name})')
            plt.xlabel('Position')
            plt.ylabel('Count of Plays')
            plt.xticks(rotation=45)
            plt.legend(title='Player Side', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(week_out, 'side_by_position.png'))
            plt.close()

        print(f"Finished {week_name}.")

    def run_all(self):
        # Look for input_*.csv files
        files = glob.glob(os.path.join(self.data_dir, "input_*.csv"))
        files.sort()
        
        if not files:
            print("No input files found!")
            return

        print(f"Found {len(files)} week files. Starting processing...")
        for f in files:
            self.analyze_week(f)

if __name__ == "__main__":
    # Define paths
    INPUT_DIR = "/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train"
    OUTPUT_DIR = "/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-analytics/visualizations/weekly_distributions"
    
    analyzer = PositionAnalyzer(INPUT_DIR, OUTPUT_DIR)
    analyzer.run_all()
