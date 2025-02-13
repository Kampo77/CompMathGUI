# Math GUI Application

This project is a graphical user interface (GUI) application that implements various mathematical methods and tasks. The application allows users to interact with different mathematical functions defined in the `comp.py` file.

## Project Structure

```
math-gui-app
├── src
│   ├── core
│   │   └── comp.py
│   └── gui
│       └── app.py
├── requirements.txt
└── README.md
```

## Description of Files

- **src/core/comp.py**: Contains various mathematical methods and tasks, including:
  - Graphical methods
  - Root-finding methods
  - Relaxation methods
  - Power methods for eigenvalues
  - Exponential curve fitting
  - Cubic spline interpolation
  - Modified Euler's method
  - Weddle's rule

- **src/gui/app.py**: Implements the graphical user interface for the application using a GUI framework such as Tkinter or PyQt. It provides windows, buttons, and input fields for user interaction.

- **requirements.txt**: Lists the dependencies required for the project, including:
  - NumPy
  - SciPy
  - Matplotlib
  - Tkinter or PyQt (for GUI)

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd math-gui-app
   ```
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

To run the application, execute the following command:
```
python src/gui/app.py
```

## Usage

Once the application is running, you can select various mathematical methods from the GUI and input the necessary parameters to see the results.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.