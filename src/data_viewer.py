
"""
This module provides a graphical user interface (GUI) application for visualizing
the NFL Big Data Bowl 2026 dataset. It allows users to select between two
data sources (prediction and analytics), choose a specific week, and load
the corresponding data into a sortable and viewable table.

Key Features:
- Interactive selection of data source (Prediction/Analytics).
- Dropdown for selecting game weeks.
- Button to load and display data for the selected week.
- Table view (Treeview) to display the loaded data, showing the first 100 rows.
- Basic error handling for data loading.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from data_reader import DataReader

class DataViewerApp:
    """
    A Tkinter application class for viewing NFL Big Data Bowl 2026 dataset.

    This class sets up the main window, UI widgets (radio buttons, combobox,
    button, and a Treeview table), and handles data loading and display logic
    using the `DataReader` class.
    """
    def __init__(self, root):
        """
        Initializes the DataViewerApp.

        Args:
            root (tk.Tk): The root Tkinter window for the application.
        """
        self.root = root
        self.root.title("NFL Big Data Bowl 2026 - Data Viewer")
        self.root.geometry("1200x600") # Set initial window size

        # --- Data Paths --- 
        # Define the directory paths for the prediction and analytics datasets.
        self.prediction_data_dir = '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-prediction/train/'
        self.analytics_data_dir = '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/'

        # --- UI Frames --- 
        # Create a frame for control widgets (dataset selection, week dropdown, load button)
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X) # Pack at the top, fill horizontally

        # Create a frame for the data display area (Treeview table)
        data_frame = ttk.Frame(self.root, padding="10")
        data_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True) # Pack at the bottom, fill and expand

        # --- Control Widgets --- 
        # Variable to hold the selected data source (prediction or analytics)
        self.data_source = tk.StringVar(value="prediction")
        
        # Radiobuttons for selecting the data source
        ttk.Label(control_frame, text="Dataset:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Radiobutton(control_frame, text="Prediction", variable=self.data_source, value="prediction", command=self.update_weeks).pack(side=tk.LEFT)
        ttk.Radiobutton(control_frame, text="Analytics", variable=self.data_source, value="analytics", command=self.update_weeks).pack(side=tk.LEFT, padx=(0, 20))

        # Label and Combobox for selecting the week
        ttk.Label(control_frame, text="Week:").pack(side=tk.LEFT, padx=(0, 5))
        self.week_var = tk.StringVar() # Variable to hold the selected week
        self.week_dropdown = ttk.Combobox(control_frame, textvariable=self.week_var, state="readonly") # 'readonly' state prevents typing
        self.week_dropdown.pack(side=tk.LEFT, padx=(0, 20))

        # Button to trigger data loading
        load_button = ttk.Button(control_frame, text="Load Data", command=self.load_data)
        load_button.pack(side=tk.LEFT)

        # --- Data Display (Treeview) --- 
        # Initialize the Treeview widget to display tabular data
        self.tree = ttk.Treeview(data_frame, show="headings") # 'show="headings"' hides the default first column
        
        # Scrollbars for the Treeview
        vsb = ttk.Scrollbar(data_frame, orient="vertical", command=self.tree.yview)
        vsb.pack(side='right', fill='y') # Pack scrollbar to the right, filling vertically
        hsb = ttk.Scrollbar(data_frame, orient="horizontal", command=self.tree.xview)
        hsb.pack(side='bottom', fill='x') # Pack scrollbar to the bottom, filling horizontally

        # Configure Treeview to use the scrollbars
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.pack(fill="both", expand=True) # Pack Treeview to fill the data_frame

        # --- Initial Population --- 
        # Call update_weeks to populate the week dropdown on application startup
        self.update_weeks()

    def update_weeks(self) -> None:
        """
        Updates the list of available weeks in the week dropdown Combobox.
        It re-initializes the DataReader based on the currently selected data source.
        Handles potential errors during data loading.
        """
        try:
            # Determine the correct data directory based on the selected radio button
            if self.data_source.get() == "prediction":
                reader = DataReader(self.prediction_data_dir)
            else: # analytics
                reader = DataReader(self.analytics_data_dir)
            
            # Get the list of weeks from the DataReader
            weeks = reader.get_weeks()
            # Update the Combobox values
            self.week_dropdown['values'] = weeks
            
            # Set the default selection to the first week if available
            if weeks:
                self.week_var.set(weeks[0])
            else:
                self.week_var.set("") # Clear selection if no weeks are found

        except Exception as e:
            # Display an error message if weeks cannot be loaded
            messagebox.showerror("Error", f"Could not load weeks: {e}")
            self.week_dropdown['values'] = [] # Clear dropdown if error occurs
            self.week_var.set("") # Clear selection

    def load_data(self) -> None:
        """
        Loads the data for the selected week and displays it in the Treeview.
        
        This method is called when the 'Load Data' button is clicked.
        It retrieves the selected week and data source, uses DataReader to
        load the data, and then calls display_dataframe to show it.
        Includes error handling for the loading process.
        """
        week = self.week_var.get() # Get the currently selected week
        
        # Validate that a week has been selected
        if not week:
            messagebox.showwarning("Warning", "Please select a week.")
            return

        try:
            # Determine the correct data directory based on the selected data source
            if self.data_source.get() == "prediction":
                reader = DataReader(self.prediction_data_dir)
            else: # analytics
                reader = DataReader(self.analytics_data_dir)
            
            # Read the combined data for the selected week
            df = reader.read_week(week)
            # Display the loaded DataFrame in the Treeview widget
            self.display_dataframe(df)

        except Exception as e:
            # Show an error message if data loading fails
            messagebox.showerror("Error", f"Could not load data for week {week}: {e}")

    def display_dataframe(self, df: pd.DataFrame) -> None:
        """
        Clears the existing data in the Treeview and populates it with new data
        from a pandas DataFrame.

        It sets up the columns and inserts the first 100 rows of the DataFrame
        into the Treeview for performance reasons.

        Args:
            df (pd.DataFrame): The DataFrame containing the data to display.
        """
        # Clear all existing items from the Treeview
        for i in self.tree.get_children():
            self.tree.delete(i)
        
        # Set the Treeview columns based on the DataFrame's columns
        self.tree["columns"] = list(df.columns)
        # Define which columns to display (all columns in this case)
        self.tree["displaycolumns"] = list(df.columns)

        # Configure column headings and properties
        for col in df.columns:
            self.tree.heading(col, text=col) # Set the header text
            # Set column properties: width, minimum width, no stretching, alignment
            self.tree.column(col, width=100, minwidth=100, stretch=tk.NO, anchor='w')

        # Insert new data into the Treeview
        # Displaying only the first 100 rows to maintain responsiveness.
        # Fill missing values (NaN) with 'N/A' for better display.
        df_display = df.head(100).fillna("N/A") 
        for index, row in df_display.iterrows(): 
            self.tree.insert("", "end", values=list(row)) # Insert row data


if __name__ == "__main__":
    # Create the main application window
    root = tk.Tk()
    # Instantiate the DataViewerApp
    app = DataViewerApp(root)
    # Start the Tkinter event loop
    root.mainloop()
