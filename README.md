# German Credit Multi-Objective Optimization

This project focuses on applying multi-objective optimization techniques to the German Credit dataset. The primary objectives are to enhance classification performance while simultaneously addressing fairness concerns and profitability. Leveraging tools like Optuna and PyMOO for hyperparameter tuning, the project aims to balance three conflicting objectives: 
1. **Accuracy**: Overall classification correctness.
2. **Fairness (DPD Score)**: Demographic Parity Difference.
3. **Expected Profit**: Financially optimal decisions using a standard cost matrix.
## Repository Structure

The repository is organized as follows:

```
German_Credit_Multi_Objective/
├── data/
│   └── ...
├── plots/
│   └── ...
├── .gitignore
├── baseline.ipynb
├── moo_solver.ipynb
├── optuna-solver.ipynb
├── pre-process.ipynb
├── README.md
├── Report.pdf
├── requirements.txt
```



* **data/**: Contains the German Credit dataset and any additional data files.
* **plots/**: Stores generated plots and visualizations from the analysis.
* **.gitignore**: Specifies files and directories to be ignored by Git.
* **baseline.ipynb**: Establishes baseline models and performance metrics.
* **moo_solver.ipynb**: Main MOO notebook evaluating 3 objectives using Optuna (NSGA-II) and PyMOO (NSGA-III) and generating visualizations.
* **optuna-solver.ipynb**: Original 2-objective Optuna formulation (Accuracy vs DPD).
* **pre-process.ipynb**: Handles data cleaning, encoding, and preprocessing steps.

## Getting Started

### Prerequisites

Ensure you have the following installed:

* Python 3.7 or higher
* Jupyter Notebook

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/AyushCodez/German_Credit_Multi_Objective.git
   cd German_Credit_Multi_Objective
   ```



2. **Create and activate a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```



3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Data Preprocessing:**

   Open and run `pre-process.ipynb` to clean and preprocess the dataset. This will prepare the data for modeling.

2. **Baseline Modeling:**

   Run `baseline.ipynb` to establish baseline performance metrics using standard classification algorithms.

3. **Hyperparameter Optimization (MOO):**

   Execute `moo_solver.ipynb` to perform hyperparameter tuning using Optuna and PyMOO. This notebook optimizes model performance across Accuracy, Fairness, and Profit. This may take a few minutes to run.

## Results

Generated plots and results from the analysis can be found in the `plots/` directory. These visualizations help in understanding the trade-offs between different objectives and the performance of various models.

![Pareto Front](plots/Pareto%20Front%20%20DPD%20vs%20Accuracy.png)
![Pareto front](plots/Pareto%20Solutions%20DPD%20VS%20Accuracy.png)

## Report
For further details regarding the motivation and implementation, please refer to [Report](Report.pdf)



