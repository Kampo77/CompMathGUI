import tkinter as tk
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.interpolate as interp
import scipy.integrate as integrate
from gui.app import TASKS  # Import TASKS from app.py

class MathApp:
    def __init__(self, master):
        self.master = master
        master.title("Computational Mathematics")
        master.geometry("800x600")
        master.configure(bg="#f0f0f0")
        
        self.mode_choice = tk.StringVar(value="manual")
        
        # Mode selection frame
        mode_frame = tk.Frame(master, bg="#f0f0f0")
        mode_frame.pack(pady=10)
        tk.Label(mode_frame, text="Mode:", font=("Helvetica", 12), bg="#f0f0f0").pack(side="left", padx=5)
        tk.Radiobutton(mode_frame, text="Manual", variable=self.mode_choice, 
                      value="manual", command=self.update_parameters, bg="#f0f0f0").pack(side="left", padx=5)
        tk.Radiobutton(mode_frame, text="Predefined", variable=self.mode_choice,
                      value="predefined", command=self.update_parameters, bg="#f0f0f0").pack(side="left", padx=5)
        
        # Method selection frame
        method_frame = tk.Frame(master, bg="#f0f0f0")
        method_frame.pack(pady=10)
        tk.Label(method_frame, text="Select Method:", font=("Helvetica", 12), bg="#f0f0f0").pack(side="left", padx=5)
        self.method_var = tk.StringVar()
        self.method_combobox = ttk.Combobox(method_frame, textvariable=self.method_var,
                                           state="readonly", values=list(TASKS.keys()))
        self.method_combobox.pack(side="left", padx=5)
        self.method_combobox.bind("<<ComboboxSelected>>", self.update_parameters)
        
        # Parameters frame
        self.params_frame = tk.Frame(master, bg="#f0f0f0")
        self.params_frame.pack(pady=10)
        self.param_entries = {}
        
        # Execute button
        self.execute_btn = tk.Button(master, text="Execute Calculation",
                                   command=self.execute_method, font=("Helvetica", 12))
        self.execute_btn.pack(pady=10)
        
        # Result display frame
        result_frame = tk.Frame(master, bg="#f0f0f0")
        result_frame.pack(pady=10, fill="both", expand=True)
        tk.Label(result_frame, text="Result:", font=("Helvetica", 12), bg="#f0f0f0").pack(anchor="w", padx=10)
        
        # ScrolledText widget for result display
        self.result_text = ScrolledText(result_frame, height=10, width=80, font=("Courier", 10))
        self.result_text.pack(padx=10, pady=5, fill="both", expand=True)
        
        # Clear button
        self.clear_btn = tk.Button(result_frame, text="Clear Result", command=self.clear_result, font=("Helvetica", 10))
        self.clear_btn.pack(pady=5)
        
        # Set initial method and parameters
        self.method_var.set(list(TASKS.keys())[0])
        self.update_parameters()
    
    def update_parameters(self, event=None):
        # Clear previous parameters
        for widget in self.params_frame.winfo_children():
            widget.destroy()
        self.param_entries = {}

        method = self.method_var.get()
        method_info = TASKS.get(method, {})
        params = method_info.get("params", [])

        # Create parameter entry fields
        for param in params:
            label_text, param_key, default = param
            lbl = tk.Label(self.params_frame, text=label_text + ":", bg="#f0f0f0",
                          font=("Helvetica", 10))
            lbl.pack(anchor="w", padx=10, pady=2)
            
            entry = tk.Entry(self.params_frame, width=50)
            entry.pack(anchor="w", padx=10, pady=2)
            
            # Set default value if in predefined mode
            if self.mode_choice.get() == "predefined":
                entry.insert(0, default)
                entry.config(state="readonly")
            
            self.param_entries[param_key] = entry
    
    def execute_method(self):
        try:
            method = self.method_var.get()
            if not method:
                messagebox.showwarning("Warning", "Please select a method!")
                return

            # Get parameters from entries
            params = {}
            for key, entry in self.param_entries.items():
                value = entry.get()
                # Try to convert to float if possible (for numerical inputs)
                try:
                    if ',' in value:  # Handle comma-separated values
                        params[key] = value
                    else:
                        params[key] = float(value) if value.replace('.', '', 1).isdigit() else value
                except ValueError:
                    params[key] = value

            # Execute the selected method
            result = TASKS[method]["function"](params)
            
            # Display the result
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, str(result))
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def clear_result(self):
        self.result_text.delete(1.0, tk.END)

def graphical_method(params):
    func_expr = params.get("Function", "x**5 - 4*x**4 + 6*x**3 - 4*x + 1")
    x_min = float(params.get("X min", 0))
    x_max = float(params.get("X max", 5))
    
    x = np.linspace(x_min, x_max, 100)
    y = eval(func_expr, {"x": x, "np": np})
    plt.figure()
    plt.plot(x, y, label="f(x)")
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.grid(True)
    plt.title("Graph of f(x)")
    plt.legend()
    plt.show()
    return "Graph displayed."

# Other methods remain unchanged...

if __name__ == "__main__":
    root = tk.Tk()
    app = MathApp(root)
    root.mainloop()