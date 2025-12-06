import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# Ensure backend acts non-interactively
import matplotlib
matplotlib.use('Agg')

class GlobalPositionAnalyzer:
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

    def load_all_data(self):
        files = glob.glob(os.path.join(self.data_dir, "input_*.csv"))
        files.sort()
        
        if not files:
            print("No input files found!")
            return None

        print(f"Found {len(files)} week files. Aggregating data...")
        
        data_frames = []
        cols_to_use = [
            'nfl_id', 'play_id', 'player_position', 'player_turnover_metrics', # Basic IDs and grouping
            'player_height', 'player_weight', # Static Attrs
            's', 'a', 'o', 'dir', # Dynamic Attrs
            'player_role', 'player_side', # Context Attrs
            'x', 'y', 'ball_land_x', 'ball_land_y' # Location Attrs
        ]
        
        # Robustly handle columns that might be missing in some files or named differently if needed
        # Based on previous file checks, these should exist. But let's be safe.
        
        for f in files:
            week = os.path.basename(f)
            print(f"Reading {week}...", end='\r')
            try:
                # Read only first row to check cols
                header = pd.read_csv(f, nrows=0).columns.tolist()
                use_cols = [c for c in cols_to_use if c in header]
                
                df_chunk = pd.read_csv(f, usecols=use_cols)
                
                # Convert height immediately to save processing later and standardise
                if 'player_height' in df_chunk.columns:
                     df_chunk['player_height_inches'] = df_chunk['player_height'].apply(self.convert_height)
                
                # Optimise types
                for col in ['player_position', 'player_role', 'player_side']:
                    if col in df_chunk.columns:
                        df_chunk[col] = df_chunk[col].astype('category')
                
                data_frames.append(df_chunk)
            except Exception as e:
                print(f"\nFailed to read {week}: {e}")

        print("\nConcatenating all weeks...")
        full_df = pd.concat(data_frames, ignore_index=True)
        print(f"Total records: {len(full_df)}")
        return full_df

    def plot_player_vs_target_locations(self, df, output_sub):
        """
        Generates scatter plots for each position comparing player location (x, y) 
        vs ball landing location (ball_land_x, ball_land_y).
        """
        if not all(col in df.columns for col in ['x', 'y', 'ball_land_x', 'ball_land_y', 'player_position']):
            print("Missing location columns. Skipping location analysis.")
            return

        print("Generating player vs target location plots...")
        loc_output_dir = os.path.join(output_sub, 'location_analysis')
        os.makedirs(loc_output_dir, exist_ok=True)

        positions = df['player_position'].unique()
        
        for pos in positions:
            if pd.isna(pos): continue
            
            # Filter data for this position
            pos_df = df[df['player_position'] == pos].copy()
            
            # Drop NaNs for plotting
            pos_df = pos_df.dropna(subset=['x', 'y', 'ball_land_x', 'ball_land_y'])
            
            if pos_df.empty:
                continue

            # Subsample if too large to prevent memory issues/overplotting
            if len(pos_df) > 50000:
                plot_data = pos_df.sample(50000, random_state=42)
            else:
                plot_data = pos_df

            plt.figure(figsize=(12, 6))
            
            # Plot Player Locations
            plt.scatter(plot_data['x'], plot_data['y'], 
                        alpha=0.2, s=5, label='Player Location', color='blue')
            
            # Plot Ball Landing Locations
            # Note: Ball landing might be the same for all players in a play, 
            # so this distribution represents the target spots relevant to this position's plays.
            plt.scatter(plot_data['ball_land_x'], plot_data['ball_land_y'], 
                        alpha=0.2, s=5, label='Ball Landing (Perfect) Location', color='red', marker='x')

            plt.title(f'Player vs Ball Landing Locations - Position: {pos}')
            plt.xlabel('X Coordinate (yards)')
            plt.ylabel('Y Coordinate (yards)')
            plt.legend()
            
            # Draw Field Borders (approximate)
            plt.axhline(0, color='gray', linestyle='--')
            plt.axhline(53.3, color='gray', linestyle='--')
            plt.axvline(0, color='gray', linestyle='--')
            plt.axvline(120, color='gray', linestyle='--')
            
            plt.tight_layout()
            plt.savefig(os.path.join(loc_output_dir, f'location_vs_target_{pos}.png'))
            plt.close()
            print(f"Generated location plot for {pos}")

    def analyze_global(self):
        df = self.load_all_data()
        if df is None or df.empty:
            print("No data to analyze.")
            return

        print("Generating global visualizations...")
        
        output_sub = self.output_dir
        os.makedirs(output_sub, exist_ok=True)

        # --- Location Analysis ---
        self.plot_player_vs_target_locations(df, output_sub)

        # --- Static Features (per player) ---
        # unique player check
        if 'nfl_id' in df.columns:
             df_player_static = df.drop_duplicates(subset=['nfl_id'])
        else:
             df_player_static = df

        # 1. Height vs Position
        if 'player_height_inches' in df_player_static.columns:
            plt.figure(figsize=(16, 8))
            sns.boxplot(x='player_position', y='player_height_inches', data=df_player_static)
            plt.title('Player Height by Position (All Weeks)')
            plt.xlabel('Position')
            plt.ylabel('Height (inches)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_sub, 'global_height_by_position.png'))
            plt.close()

        # 2. Weight vs Position
        if 'player_weight' in df_player_static.columns:
            plt.figure(figsize=(16, 8))
            sns.boxplot(x='player_position', y='player_weight', data=df_player_static)
            plt.title('Player Weight by Position (All Weeks)')
            plt.xlabel('Position')
            plt.ylabel('Weight (lbs)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_sub, 'global_weight_by_position.png'))
            plt.close()

        # --- Dynamic Features (per frame) ---
        # 3. Speed vs Position
        if 's' in df.columns:
            plt.figure(figsize=(16, 8))
            sns.boxplot(x='player_position', y='s', data=df)
            plt.title('Speed by Position (All Weeks)')
            plt.xlabel('Position')
            plt.ylabel('Speed (yards/s)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_sub, 'global_speed_by_position.png'))
            plt.close()

        # 4. Acceleration vs Position
        if 'a' in df.columns:
            plt.figure(figsize=(16, 8))
            sns.boxplot(x='player_position', y='a', data=df)
            plt.title('Acceleration by Position (All Weeks)')
            plt.xlabel('Position')
            plt.ylabel('Acceleration (yards/s^2)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_sub, 'global_acceleration_by_position.png'))
            plt.close()
        
        # 5. Orientation (o) vs Position
        if 'o' in df.columns:
            plt.figure(figsize=(16, 8))
            sns.boxplot(x='player_position', y='o', data=df)
            plt.title('Orientation (o) by Position (All Weeks)')
            plt.xlabel('Position')
            plt.ylabel('Orientation (degrees)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_sub, 'global_orientation_by_position.png'))
            plt.close()

        # 6. Direction (dir) vs Position
        if 'dir' in df.columns:
            plt.figure(figsize=(16, 8))
            sns.boxplot(x='player_position', y='dir', data=df)
            plt.title('Direction (dir) by Position (All Weeks)')
            plt.xlabel('Position')
            plt.ylabel('Direction (degrees)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_sub, 'global_direction_by_position.png'))
            plt.close()

        # --- Categorical/Play Features ---
        # Drop duplicates for role/side analysis per play per player to avoid frame bias
        if 'play_id' in df.columns and 'nfl_id' in df.columns:
            df_play_static = df.drop_duplicates(subset=['play_id', 'nfl_id'])
        else:
            df_play_static = df

        # 7. Role Distribution
        if 'player_role' in df.columns:
            plt.figure(figsize=(16, 8))
            ct = pd.crosstab(df_play_static['player_position'], df_play_static['player_role'])
            ct.plot(kind='bar', stacked=True, figsize=(16, 8), cmap='viridis')
            plt.title('Player Role Distribution by Position (All Weeks)')
            plt.xlabel('Position')
            plt.ylabel('Count of Plays')
            plt.xticks(rotation=45)
            plt.legend(title='Player Role', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(output_sub, 'global_role_by_position.png'))
            plt.close()

        # 8. Side Distribution
        if 'player_side' in df.columns:
            plt.figure(figsize=(16, 8))
            ct = pd.crosstab(df_play_static['player_position'], df_play_static['player_side'])
            ct.plot(kind='bar', stacked=True, figsize=(16, 8), cmap='Set2')
            plt.title('Player Side Distribution by Position (All Weeks)')
            plt.xlabel('Position')
            plt.ylabel('Count of Plays')
            plt.xticks(rotation=45)
            plt.legend(title='Player Side', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(output_sub, 'global_side_by_position.png'))
            plt.close()

        print("Global analysis complete.")

if __name__ == "__main__":
    INPUT_DIR = "/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train"
    OUTPUT_DIR = "/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-analytics/visualizations/global_distributions"
    
    analyzer = GlobalPositionAnalyzer(INPUT_DIR, OUTPUT_DIR)
    analyzer.analyze_global()
