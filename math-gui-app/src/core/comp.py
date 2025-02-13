import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.interpolate as interp
import scipy.integrate as integrate

class MathApp:
    def __init__(self, master):
        self.master = master
        master.title("Computational Mathematics")
        master.geometry("700x600")
        
        self.mode_choice = tk.StringVar(value="manual")
        
        mode_frame = tk.Frame(master)
        mode_frame.pack(pady=5)
        tk.Label(mode_frame, text="Mode:").pack(side="left", padx=5)
        tk.Radiobutton(mode_frame, text="Manual", variable=self.mode_choice, 
                      value="manual", command=self.update_parameters).pack(side="left", padx=5)
        tk.Radiobutton(mode_frame, text="Predefined", variable=self.mode_choice,
                      value="predefined", command=self.update_parameters).pack(side="left", padx=5)
        
        self.method_var = tk.StringVar()
        self.method_combobox = ttk.Combobox(master, textvariable=self.method_var,
                                           state="readonly", values=list(TASKS.keys()))
        self.method_combobox.pack(pady=5)
        self.method_combobox.bind("<<ComboboxSelected>>", self.update_parameters)
        
        self.params_frame = tk.Frame(master)
        self.params_frame.pack(pady=10)
        self.param_entries = {}
        
        self.execute_btn = tk.Button(master, text="Execute Calculation",
                                   command=self.execute_method, font=("Helvetica", 12))
        self.execute_btn.pack(pady=10)
        
        self.result_text = tk.Text(master, height=10, width=80)
        self.result_text.pack(pady=5)
        
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

def root_finding_methods(params):
    func = lambda x: np.log(x) - x/10
    root_false_position = opt.root_scalar(func, bracket=[0.01, 10], method='bisect')
    root_newton = opt.newton(func, x0=1.0)
    iterations_false = root_false_position.iterations
    iterations_newton = root_false_position.function_calls
    return f"False Position Method: Root = {root_false_position.root:.6f} (iterations: {iterations_false})\nNewton's Method: Root = {root_newton:.6f} (iterations: {iterations_newton})"

def relaxation_method(params):
    matrix_str = params.get("Matrix", "1,1,1,9;2,-3,4,13;3,4,5,40")
    matrix_rows = matrix_str.split(';')
    A = np.array([list(map(float, row.split(','))) for row in matrix_rows])
    b = A[:, -1]
    A = A[:, :-1]
    
    omega = 0.9
    x = np.zeros(len(b))
    max_iter = 100
    tol = 1e-6
    
    for iter in range(max_iter):
        x_old = x.copy()
        for i in range(len(b)):
            sum1 = np.dot(A[i, :i], x[:i])
            sum2 = np.dot(A[i, i+1:], x_old[i+1:])
            x[i] = (1 - omega) * x_old[i] + (omega / A[i, i]) * (b[i] - sum1 - sum2)
        
        if np.allclose(x, x_old, rtol=tol):
            break
    
    return f"Solution: {x}\nIterations: {iter+1}"

def power_method(params):
    matrix_str = params.get("Matrix", "2,1;1,3")
    matrix_rows = matrix_str.split(';')
    A = np.array([list(map(float, row.split(','))) for row in matrix_rows])
    
    max_iter = 100
    tol = 1e-6
    x = np.ones(len(A))
    
    for i in range(max_iter):
        x_new = A @ x
        eigenvalue = np.max(np.abs(x_new))
        x_new = x_new / eigenvalue
        if np.allclose(x, x_new, rtol=tol):
            break
        x = x_new
    
    return f"Largest eigenvalue: {eigenvalue:.6f}\nIterations: {i+1}"

def exponential_curve_fitting(params):
    x_str = params.get("X values", "0.5,1.5,2.5,3.5")
    y_str = params.get("Y values", "2,6,18,54")
    
    x = np.array([float(val) for val in x_str.split(',')])
    y = np.array([float(val) for val in y_str.split(',')])
    
    log_y = np.log(y)
    coeffs = np.polyfit(x, log_y, 1)
    a = np.exp(coeffs[1])
    b = coeffs[0]
    
    plt.figure()
    plt.scatter(x, y, label='Data points')
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = a * np.exp(b * x_fit)
    plt.plot(x_fit, y_fit, 'r-', label=f'Fit: y = {a:.2f}e^({b:.2f}x)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return f"Fitted equation: y = {a:.2f}e^({b:.2f}x)"

def cubic_spline_interpolation(params):
    x_str = params.get("X values", "0.5,1.5,2.5,3.5")
    y_str = params.get("Y values", "0.25,0.75,2.25,6.25")
    
    x = np.array([float(val) for val in x_str.split(',')])
    y = np.array([float(val) for val in y_str.split(',')])
    
    cs = interp.CubicSpline(x, y)
    x_new = np.linspace(min(x), max(x), 100)
    y_new = cs(x_new)
    
    plt.figure()
    plt.scatter(x, y, label='Data points')
    plt.plot(x_new, y_new, 'r-', label='Cubic spline')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return "Cubic spline interpolation completed"

def modified_euler(params):
    x0 = float(params.get("Initial x", "0"))
    y0 = float(params.get("Initial y", "1"))
    h = float(params.get("Step size", "0.2"))
    steps = int(params.get("Number of steps", "2"))
    
    def f(x, y):
        return np.sin(x) - y
    
    x = [x0]
    y = [y0]
    
    for i in range(steps):
        k1 = f(x[-1], y[-1])
        k2 = f(x[-1] + h, y[-1] + h*k1)
        y_new = y[-1] + h/2 * (k1 + k2)
        x.append(x[-1] + h)
        y.append(y_new)
    
    return f"y({x[-1]}) = {y[-1]}"

def weddles_rule(params):
    func_expr = params.get("Function", "1/(1+x**2)")
    interval_str = params.get("Interval (a,b)", "0,6")
    a, b = map(float, interval_str.split(','))
    
    n = 6  # number of subintervals
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = np.array([eval(func_expr, {"x": val, "np": np}) for val in x])
    
    result = (3*h/10) * (y[0] + y[6] + 5*y[1] + y[2] + 6*y[3] + y[4] + 5*y[5])
    exact_result, _ = integrate.quad(lambda x: eval(func_expr, {"x": x, "np": np}), a, b)
    
    return f"Weddle's Rule result: {result:.6f}\nExact result: {exact_result:.6f}\nAbsolute error: {abs(result-exact_result):.6f}"

TASKS = {
    "Task 1: Graphical Method": {
        "function": graphical_method,
        "params": [
            ("Function (in terms of x)", "Function", "x**5 - 4*x**4 + 6*x**3 - 4*x + 1"),
            ("X min", "X min", "0"),
            ("X max", "X max", "5")
        ]
    },
    "Task 2: Root-Finding Methods": {
        "function": root_finding_methods,
        "params": [("Coefficients (comma-separated)", "Coefficients", "0.01,10")]
    },
    "Task 3: Relaxation Method": {
        "function": relaxation_method,
        "params": [("Matrix (rows separated by ';', columns by ',')", "Matrix", "1,1,1,9;2,-3,4,13;3,4,5,40")]
    },
    "Task 4: Power Method": {
        "function": power_method,
        "params": [("Matrix (rows separated by ';', columns by ',')", "Matrix", "2,1;1,3")]
    },
    "Task 5: Exponential Curve Fitting": {
        "function": exponential_curve_fitting,
        "params": [
            ("X values (comma-separated)", "X values", "0.5,1.5,2.5,3.5"),
            ("Y values (comma-separated)", "Y values", "2,6,18,54")
        ]
    },
    "Task 6: Cubic Spline Interpolation": {
        "function": cubic_spline_interpolation,
        "params": [
            ("X values (comma-separated)", "X values", "0.5,1.5,2.5,3.5"),
            ("Y values (comma-separated)", "Y values", "0.25,0.75,2.25,6.25")
        ]
    },
    "Task 7: Modified Euler": {
        "function": modified_euler,
        "params": [
            ("Initial x", "Initial x", "0"),
            ("Initial y", "Initial y", "1"),
            ("Step size", "Step size", "0.2"),
            ("Number of steps", "Number of steps", "2")
        ]
    },
    "Task 8: Weddle's Rule": {
        "function": weddles_rule,
        "params": [
            ("Function (in terms of x)", "Function", "1/(1+x**2)"),
            ("Interval (a,b)", "Interval (a,b)", "0,6")
        ]
    }
}

if __name__ == "__main__":
    root = tk.Tk()
    app = MathApp(root)
    root.mainloop()
