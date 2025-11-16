import tkinter as tk
from tkinter import ttk, messagebox
from data_reader import DataReader

class DataViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NFL Big Data Bowl 2026 - Data Viewer")
        self.root.geometry("1200x600")

        # --- Data Paths ---
        self.prediction_data_dir = '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-prediction/train/'
        self.analytics_data_dir = '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/'

        # --- UI Frames ---
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)

        data_frame = ttk.Frame(self.root, padding="10")
        data_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # --- Control Widgets ---
        self.data_source = tk.StringVar(value="prediction")
        
        ttk.Label(control_frame, text="Dataset:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Radiobutton(control_frame, text="Prediction", variable=self.data_source, value="prediction", command=self.update_weeks).pack(side=tk.LEFT)
        ttk.Radiobutton(control_frame, text="Analytics", variable=self.data_source, value="analytics", command=self.update_weeks).pack(side=tk.LEFT, padx=(0, 20))

        ttk.Label(control_frame, text="Week:").pack(side=tk.LEFT, padx=(0, 5))
        self.week_var = tk.StringVar()
        self.week_dropdown = ttk.Combobox(control_frame, textvariable=self.week_var, state="readonly")
        self.week_dropdown.pack(side=tk.LEFT, padx=(0, 20))

        load_button = ttk.Button(control_frame, text="Load Data", command=self.load_data)
        load_button.pack(side=tk.LEFT)

        # --- Data Display (Treeview) ---
        self.tree = ttk.Treeview(data_frame, show="headings")
        
        vsb = ttk.Scrollbar(data_frame, orient="vertical", command=self.tree.yview)
        vsb.pack(side='right', fill='y')
        hsb = ttk.Scrollbar(data_frame, orient="horizontal", command=self.tree.xview)
        hsb.pack(side='bottom', fill='x')

        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.pack(fill="both", expand=True)

        # --- Initial Population ---
        self.update_weeks()

    def update_weeks(self):
        try:
            if self.data_source.get() == "prediction":
                reader = DataReader(self.prediction_data_dir)
            else:
                reader = DataReader(self.analytics_data_dir)
            
            weeks = reader.get_weeks()
            self.week_dropdown['values'] = weeks
            if weeks:
                self.week_var.set(weeks[0])
        except Exception as e:
            messagebox.showerror("Error", f"Could not load weeks: {e}")
            self.week_dropdown['values'] = []
            self.week_var.set("")


    def load_data(self):
        week = self.week_var.get()
        if not week:
            messagebox.showwarning("Warning", "Please select a week.")
            return

        try:
            if self.data_source.get() == "prediction":
                reader = DataReader(self.prediction_data_dir)
            else:
                reader = DataReader(self.analytics_data_dir)
            
            df = reader.read_week(week)
            self.display_dataframe(df)

        except Exception as e:
            messagebox.showerror("Error", f"Could not load data for week {week}: {e}")

    def display_dataframe(self, df):
        # Clear previous data
        for i in self.tree.get_children():
            self.tree.delete(i)
        
        # Set new columns
        self.tree["columns"] = list(df.columns)
        self.tree["displaycolumns"] = list(df.columns)

        for col in df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, minwidth=100, stretch=tk.NO, anchor='w')

        # Insert new data
        df_display = df.head(100).fillna("N/A")
        for index, row in df_display.iterrows(): # Displaying first 100 rows for performance
            self.tree.insert("", "end", values=list(row))


if __name__ == "__main__":
    root = tk.Tk()
    app = DataViewerApp(root)
    root.mainloop()
