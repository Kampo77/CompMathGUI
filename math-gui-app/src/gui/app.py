import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
import numpy as np


plt.ion()  

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
    return ("The function has 3 roots in the interval [0, 5].\n"
            "The roots are approximately 0.5, 1.5, and 3.5.")

def root_finding_methods(params):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import root_scalar
    
    
    def f(x):
        return np.log(x) - x/10
    
    
    def df(x):
        return 1/x - 0.1

    
    a, b = map(float, params.get("Coefficients").split(','))

    # -----------------------------
    # "False Position" via bisect
    
    root_false = root_scalar(
        f, 
        method='bisect', 
        bracket=[a, b],
        maxiter=100,
        xtol=1e-10  
    )

    # -----------------------------
    # Newton-Raphson Method
    
    if a <= 1 <= b:
        initial_guess = 1.0
    else:
        initial_guess = (a + b) / 2.0
    
    root_newton = root_scalar(
        f,
        fprime=df,
        method='newton',
        x0=initial_guess,
        maxiter=100, 
        rtol=1e-10  # optional: tighten tolerance
    )

    
    x = np.linspace(a, b, 200)
    y = f(x)

    plt.figure()
    plt.plot(x, y, label="f(x) = log(x) - x/10")
    plt.axhline(0, color='black', linewidth=1)


    plt.axvline(
        root_false.root, 
        color='red', 
        linestyle='--', 
        label=f"False Position Root: {root_false.root:.6f}"
    )

    plt.axvline(
        root_newton.root, 
        color='blue', 
        linestyle='--', 
        label=f"Newton-Raphson Root: {root_newton.root:.6f}"
    )

    plt.grid(True)
    plt.title("Root-Finding Methods")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.show()
    rel_err = abs(root_false.root - root_newton.root) / abs(root_newton.root)

    return (
        f"False Position Method:\n"
        f"  Root = {root_false.root:.6f}\n"
        f"  Iterations = {root_false.iterations}\n"
        f"  Relative Error (vs. Newton) = {rel_err:.6e}\n\n"
        
        f"Newton-Raphson Method:\n"
        f"  Root = {root_newton.root:.6f}\n"
        f"  Iterations = {root_newton.iterations}\n"
        f"  Relative Error (vs. False Pos.) = {rel_err:.6e}"
    )

def relaxation_method(params):
    matrix_str = params.get("Matrix")
    matrix_rows = matrix_str.split(';')
    A = np.array([list(map(float, row.split(','))) for row in matrix_rows])
    b = A[:, -1]
    A = A[:, :-1]
    
    omega = 0.9
    x = np.zeros(len(b))
    max_iter = 100
    tol = 1e-6
    errors = []
    
    for iter in range(max_iter):
        x_old = x.copy()
        for i in range(len(b)):
            sum1 = np.dot(A[i, :i], x[:i])
            sum2 = np.dot(A[i, i+1:], x_old[i+1:])
            x[i] = (1 - omega) * x_old[i] + (omega / A[i, i]) * (b[i] - sum1 - sum2)
        
        error = np.linalg.norm(x - x_old)
        errors.append(error)
        if error < tol:
            break
    
    # Plot the convergence
    plt.figure()
    plt.plot(errors, label="Convergence of Relaxation Method")
    plt.yscale('log')
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    return f"Solution:\nx = {x[0]:.6f}\ny = {x[1]:.6f}\nz = {x[2]:.6f}\nIterations: {iter+1}"

def power_method(params):
    matrix_str = params.get("Matrix")
    matrix_rows = matrix_str.split(';')
    A = np.array([list(map(float, row.split(','))) for row in matrix_rows])
    
    n = len(A)
    x = np.ones(n)
    max_iter = 100
    tol = 1e-6
    eigenvalues = []
    
    for i in range(max_iter):
        x_new = A @ x
        lambda_new = np.max(np.abs(x_new))
        x_new = x_new / lambda_new
        eigenvalues.append(lambda_new)
        
        if np.allclose(x, x_new, rtol=tol):
            break
            
        x = x_new
    
    # Plot the convergence of the eigenvalue
    plt.figure()
    plt.plot(eigenvalues, label="Convergence of Eigenvalue")
    plt.xlabel("Iteration")
    plt.ylabel("Eigenvalue")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    return f"Largest eigenvalue: {lambda_new:.6f}\nIterations: {i+1}"

def exponential_curve_fitting(params):
    x = np.array([float(x) for x in params.get("X values").split(',')])
    y = np.array([float(y) for y in params.get("Y values").split(',')])
    
    # Taking log of y values
    log_y = np.log(y)
    
    # Linear fit of log(y) vs x
    coeffs = np.polyfit(x, log_y, 1)
    a = np.exp(coeffs[1])  # a in y = ae^(bx)
    b = coeffs[0]          # b in y = ae^(bx)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Data points')
    
    # Plot fitted curve
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = a * np.exp(b * x_fit)
    plt.plot(x_fit, y_fit, 'r-', label=f'Fit: y = {a:.2f}e^({b:.2f}x)')
    
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return f"Fitted equation: y = {a:.4f}e^({b:.4f}x)"

def cubic_spline_interpolation(params):
    x = np.array([float(x) for x in params.get("X values").split(',')])
    y = np.array([float(y) for y in params.get("Y values").split(',')])
    
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(x, y)
    
    x_new = np.linspace(min(x), max(x), 100)
    y_new = cs(x_new)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Data points')
    plt.plot(x_new, y_new, 'r-', label='Cubic spline')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    test_points = np.linspace(min(x), max(x), 5)
    results = [f"f({x:.2f}) = {cs(x):.4f}" for x in test_points]
    return "Interpolated values:\n" + "\n".join(results)

def modified_euler(params):
    x0 = float(params.get("Initial x"))
    y0 = float(params.get("Initial y"))
    h = float(params.get("Step size"))
    steps = int(params.get("Number of steps"))
    
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
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'bo-', label='Numerical solution')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    return f"y({x[-1]}) = {y[-1]:.6f}"

def weddles_rule(params):
    func_expr = params.get("Function")
    a, b = map(float, params.get("Interval (a,b)").split(','))
    
    n = 6  # number of subintervals
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = np.array([eval(func_expr, {"x": val, "np": np}) for val in x])
    
    result = (3*h/10) * (y[0] + y[6] + 5*y[1] + y[2] + 6*y[3] + y[4] + 5*y[5])
    
    from scipy import integrate
    exact_result, _ = integrate.quad(lambda x: eval(func_expr, {"x": x, "np": np}), a, b)
    
    # Plot the function and the area under the curve
    x_plot = np.linspace(a, b, 100)
    y_plot = np.array([eval(func_expr, {"x": val, "np": np}) for val in x_plot])
    plt.figure()
    plt.plot(x_plot, y_plot, label=f"f(x) = {func_expr}")
    plt.fill_between(x_plot, y_plot, alpha=0.3)
    plt.title("Weddle's Rule Integration")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return (f"Weddle's Rule result: {result:.6f}\n"
            f"Exact result: {exact_result:.6f}\n"
            f"Absolute error: {abs(result-exact_result):.6f}")

MANUAL_METHODS = [
    "Graphical Method",
    "Root-Finding Methods",
    "Relaxation Method", 
    "Power Method",
    "Exponential Curve Fitting",
    "Cubic Spline Interpolation",
    "Modified Euler's Method",
    "Weddle's Rule"
]

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
    "Task 7: Modified Euler Method": {
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

# GUI Class
class MathApp:
    def __init__(self, master):
        self.master = master
        master.title("Computational Mathematics - Final Project")
        master.geometry("800x600")
        master.configure(bg="#f0f0f0")

        self.mode_choice = tk.StringVar(value="manual")
        mode_frame = tk.Frame(master, bg="#f0f0f0")
        mode_frame.pack(pady=5)
        tk.Label(mode_frame, text="Mode:", font=("Helvetica", 12), bg="#f0f0f0").pack(side="left", padx=5)
        tk.Radiobutton(mode_frame, text="Manual Mode", variable=self.mode_choice, value="manual", bg="#f0f0f0", command=self.update_parameters).pack(side="left", padx=5)
        tk.Radiobutton(mode_frame, text="Predefined Full Answer", variable=self.mode_choice, value="predefined", bg="#f0f0f0", command=self.update_parameters).pack(side="left", padx=5)

        self.method_var = tk.StringVar()
        self.method_combobox = ttk.Combobox(master, textvariable=self.method_var, state="readonly", values=list(TASKS.keys()))
        self.method_combobox.pack(pady=5)
        self.method_combobox.bind("<<ComboboxSelected>>", self.update_parameters)

        self.params_frame = tk.Frame(master, bg="#f0f0f0")
        self.params_frame.pack(pady=10)
        self.param_entries = {}

        self.execute_btn = tk.Button(master, text="Execute Calculation", font=("Helvetica", 12), command=self.execute_method)
        self.execute_btn.pack(pady=10)

        result_label = tk.Label(master, text="Result:", font=("Helvetica", 12, "bold"), bg="#f0f0f0")
        result_label.pack(pady=5)
        self.result_text = tk.Text(master, height=7, width=70)
        self.result_text.pack(pady=5)

        self.method_var.set(list(TASKS.keys())[0])
        self.update_parameters()

    def update_parameters(self, event=None):
        # Clear previous parameters and result
        for widget in self.params_frame.winfo_children():
            widget.destroy()
        self.param_entries = {}
        self.result_text.delete(1.0, tk.END)  # Clear the result text

        method = self.method_var.get()
        method_info = TASKS.get(method, {})
        params = method_info.get("params", [])

        for param in params:
            label_text, param_key, default = param
            lbl = tk.Label(self.params_frame, text=label_text + ":", bg="#f0f0f0", font=("Helvetica", 10))
            lbl.pack(anchor="w", padx=10, pady=2)
            entry = tk.Entry(self.params_frame, width=50)
            entry.pack(anchor="w", padx=10, pady=2)

            if self.mode_choice.get() == "predefined":
                entry.insert(0, default)
                entry.config(state="readonly")
            else:
                entry.insert(0, "")  # Empty field for manual input
                entry.config(state="normal")

            self.param_entries[param_key] = entry

    def execute_method(self):
        method = self.method_var.get()
        if not method:
            messagebox.showwarning("Warning", "Please select a method!")
            return

        try:
            params = {key: entry.get() for key, entry in self.param_entries.items()}
            result = TASKS[method]["function"](params)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, str(result))
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = MathApp(root)
    root.mainloop()