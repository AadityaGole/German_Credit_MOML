import nbformat as nbf

nb = nbf.v4.new_notebook()

text = """# Multi-Objective Optimization for German Credit Risk
This notebook implements a Multi-Objective Optimization (MOO) pipeline to optimize a credit risk model (XGBoost) across three conflicting objectives:
1. **Accuracy (Maximize)**: Overall classification correctness.
2. **Fairness (Minimize DPD)**: Demographic Parity Difference between male and female groups.
3. **Expected Profit (Maximize)**: Based on an asymmetric cost matrix (+1 for correct approval/rejection, -5 for incorrect approval/default).

We implement and compare two MOO algorithms:
- **Optuna NSGA-II**
- **PyMOO NSGA-III**
"""

code_setup = """import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import optuna
import plotly.express as px
import plotly.graph_objects as go
import random
import os
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
os.makedirs("plots", exist_ok=True)

GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# Load data
DATA_PATH = 'data/german_credit_data_cleaned.csv'
df = pd.read_csv(DATA_PATH)
y = df['Risk_good'].values
X_full = df.drop(columns=['Risk_good'])
sensitive = df['Sex_male'].values
"""

text_obj = """## Custom Objective and Evaluation Functions
We use a custom objective function during XGBoost training to incorporate a fairness penalty (controlled by `alpha_fair`). The classification threshold is also a decision variable.
"""

code_obj = """def make_custom_obj(sens_array, alpha_fair=0.5):
    '''Custom XGBoost objective combining log-loss and fairness penalty.'''
    def custom_obj(preds, dmatrix):
        labels = dmatrix.get_label()
        preds_prob = 1.0 / (1.0 + np.exp(-preds))
        
        # Log-loss gradients
        grad_log = preds_prob - labels
        hess_log = preds_prob * (1.0 - preds_prob)
        
        # Fairness gradients (Demographic Parity)
        mask_p = (sens_array == 1)
        mask_u = (sens_array == 0)
        n_p, n_u = mask_p.sum(), mask_u.sum()
        
        mean_p = preds_prob[mask_p].mean() if n_p > 0 else 0
        mean_u = preds_prob[mask_u].mean() if n_u > 0 else 0
        diff = mean_p - mean_u
        sign = np.sign(diff)
        
        sigmoid_deriv = preds_prob * (1.0 - preds_prob)
        grad_fair = np.zeros_like(preds)
        
        if n_p > 0: grad_fair[mask_p] = sign * (1.0 / n_p) * sigmoid_deriv[mask_p]
        if n_u > 0: grad_fair[mask_u] = -sign * (1.0 / n_u) * sigmoid_deriv[mask_u]
        
        hess_fair = np.zeros_like(preds) # simplified
        
        grad = (1 - alpha_fair) * grad_log + alpha_fair * grad_fair
        hess = (1 - alpha_fair) * hess_log + alpha_fair * hess_fair
        return grad, hess
    return custom_obj

def evaluate_model(params, alpha_fair, threshold, seed=GLOBAL_SEED):
    '''Evaluates a configuration using 3-fold CV. Returns (accuracy, dpd, profit).'''
    kf = KFold(n_splits=3, shuffle=True, random_state=seed)
    acc_scores, dpd_scores, profit_scores = [], [], []

    for train_idx, val_idx in kf.split(X_full):
        X_train = X_full.iloc[train_idx].drop(columns=['Sex_male']).values
        y_train, sens_train = y[train_idx], sensitive[train_idx]
        
        X_val = X_full.iloc[val_idx].drop(columns=['Sex_male']).values
        y_val, sens_val = y[val_idx], sensitive[val_idx]

        train_obj = make_custom_obj(sens_train, alpha_fair=alpha_fair)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        bst = xgb.train(
            params, dtrain, num_boost_round=100, obj=train_obj,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=10, verbose_eval=False
        )

        pred_prob = bst.predict(dval)
        pred = (pred_prob > threshold).astype(int)

        # 1. Accuracy
        acc = accuracy_score(y_val, pred)
        
        # 2. Fairness (DPD)
        dpd = abs(pred[sens_val == 1].mean() - pred[sens_val == 0].mean())
        
        # 3. Expected Profit
        # Cost matrix: True Positive (+1), True Negative (+1), False Positive/Default (-5), False Negative (0)
        profit = 0
        for i in range(len(y_val)):
            if y_val[i] == 1 and pred[i] == 1: profit += 1 # Good loan approved
            elif y_val[i] == 0 and pred[i] == 0: profit += 1 # Bad loan rejected
            elif y_val[i] == 0 and pred[i] == 1: profit -= 5 # Bad loan approved (DEFAULT)
            # Good loan rejected = 0 (missed opportunity)
        
        avg_profit = profit / len(y_val)

        acc_scores.append(acc)
        dpd_scores.append(dpd)
        profit_scores.append(avg_profit)

    return np.mean(acc_scores), np.mean(dpd_scores), np.mean(profit_scores)
"""

text_optuna = """## Algorithm 1: Optuna NSGA-II
We use Optuna's `NSGAIISampler` to optimize the 3 objectives over 200 trials.
"""

code_optuna = """def optuna_objective(trial):
    max_depth = trial.suggest_int('max_depth', 3, 7)
    eta = trial.suggest_float('eta', 0.01, 0.3)
    subsample = trial.suggest_float('subsample', 0.6, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0)
    
    alpha_fair = trial.suggest_float('alpha_fair', 0.0, 1.0)
    threshold = trial.suggest_float('threshold', 0.3, 0.7)
    
    params = {
        'max_depth': max_depth,
        'eta': eta,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'verbosity': 0,
        'seed': GLOBAL_SEED + trial.number
    }
    
    acc, dpd, profit = evaluate_model(params, alpha_fair, threshold)
    
    trial.set_user_attr('params', params)
    trial.set_user_attr('alpha_fair', alpha_fair)
    trial.set_user_attr('threshold', threshold)
    
    # Optuna maximizes accuracy, minimizes DPD, maximizes profit
    return acc, dpd, profit

# Run Optuna Study
sampler = optuna.samplers.NSGAIISampler(seed=GLOBAL_SEED)
study = optuna.create_study(directions=['maximize', 'minimize', 'maximize'], sampler=sampler)
study.optimize(optuna_objective, n_trials=200, show_progress_bar=True)

# Extract Pareto front
optuna_results = []
for trial in study.best_trials:
    optuna_results.append({
        'Algorithm': 'Optuna NSGA-II',
        'Accuracy': trial.values[0],
        'DPD': trial.values[1],
        'Profit': trial.values[2],
        'alpha_fair': trial.user_attrs['alpha_fair'],
        'threshold': trial.user_attrs['threshold']
    })
df_optuna = pd.DataFrame(optuna_results)
print(f"Optuna found {len(df_optuna)} Pareto-optimal solutions.")
"""

text_pymoo = """## Algorithm 2: PyMOO NSGA-III
We define an `ElementwiseProblem` for PyMOO and run NSGA-III with Das-Dennis reference directions.
"""

code_pymoo = """class CreditRiskProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=6, n_obj=3, n_ieq_constr=0,
                         xl=np.array([3, 0.01, 0.6, 0.6, 0.0, 0.3]),
                         xu=np.array([7, 0.3, 1.0, 1.0, 1.0, 0.7]))

    def _evaluate(self, x, out, *args, **kwargs):
        max_depth = int(np.round(x[0]))
        eta = x[1]
        subsample = x[2]
        colsample_bytree = x[3]
        alpha_fair = x[4]
        threshold = x[5]

        params = {
            'max_depth': max_depth,
            'eta': eta,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'verbosity': 0,
            'seed': GLOBAL_SEED
        }

        acc, dpd, profit = evaluate_model(params, alpha_fair, threshold)

        # PyMOO minimizes all objectives by default
        # Maximize accuracy -> minimize -accuracy
        # Minimize DPD -> minimize DPD
        # Maximize profit -> minimize -profit
        out["F"] = [-acc, dpd, -profit]

# Run PyMOO Study
ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
algorithm = NSGA3(pop_size=100, ref_dirs=ref_dirs)
problem = CreditRiskProblem()

res = minimize(problem,
               algorithm,
               ('n_gen', 50),
               seed=GLOBAL_SEED,
               verbose=True)

# Extract Pareto front
pymoo_results = []
if res.F is not None:
    for i in range(len(res.F)):
        pymoo_results.append({
            'Algorithm': 'PyMOO NSGA-III',
            'Accuracy': -res.F[i][0],
            'DPD': res.F[i][1],
            'Profit': -res.F[i][2],
            'alpha_fair': res.X[i][4],
            'threshold': res.X[i][5]
        })
df_pymoo = pd.DataFrame(pymoo_results)
print(f"PyMOO found {len(df_pymoo)} Pareto-optimal solutions.")
"""

text_combine = """## Analysis and Visualizations
We combine the results from both algorithms and visualize the trade-offs.
"""

code_combine = """# Combine results
df_all = pd.concat([df_optuna, df_pymoo], ignore_index=True)
df_all.to_csv('data/moo_pareto_fronts.csv', index=False)

# 1. 3D Pareto Front
fig = px.scatter_3d(
    df_all, x='DPD', y='Accuracy', z='Profit',
    color='Algorithm', symbol='Algorithm',
    hover_data=['alpha_fair', 'threshold'],
    title='3D Pareto Front: Optuna NSGA-II vs PyMOO NSGA-III'
)
fig.update_layout(scene=dict(
    xaxis_title='DPD (Minimize)',
    yaxis_title='Accuracy (Maximize)',
    zaxis_title='Expected Profit (Maximize)'
))
fig.write_html("plots/3d_pareto_front.html")
fig.show()

# 2. Parallel Coordinates
fig2 = px.parallel_coordinates(
    df_all,
    dimensions=['alpha_fair', 'threshold', 'DPD', 'Accuracy', 'Profit'],
    color='Profit',
    title='Parallel Coordinates Plot of Pareto Solutions'
)
fig2.write_html("plots/parallel_coordinates.html")
fig2.show()

# 3. 2D Projections
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for alg in ['Optuna NSGA-II', 'PyMOO NSGA-III']:
    subset = df_all[df_all['Algorithm'] == alg]
    
    axes[0].scatter(subset['DPD'], subset['Accuracy'], label=alg, alpha=0.7)
    axes[0].set_xlabel('DPD (lower is better)'); axes[0].set_ylabel('Accuracy (higher is better)')
    
    axes[1].scatter(subset['DPD'], subset['Profit'], label=alg, alpha=0.7)
    axes[1].set_xlabel('DPD (lower is better)'); axes[1].set_ylabel('Expected Profit (higher is better)')
    
    axes[2].scatter(subset['Accuracy'], subset['Profit'], label=alg, alpha=0.7)
    axes[2].set_xlabel('Accuracy (higher is better)'); axes[2].set_ylabel('Expected Profit (higher is better)')

axes[0].legend(); axes[1].legend(); axes[2].legend()
plt.suptitle('2D Projections of the Pareto Front')
plt.tight_layout()
plt.savefig('plots/2d_projections.png', dpi=150)
plt.show()
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text),
    nbf.v4.new_code_cell(code_setup),
    nbf.v4.new_markdown_cell(text_obj),
    nbf.v4.new_code_cell(code_obj),
    nbf.v4.new_markdown_cell(text_optuna),
    nbf.v4.new_code_cell(code_optuna),
    nbf.v4.new_markdown_cell(text_pymoo),
    nbf.v4.new_code_cell(code_pymoo),
    nbf.v4.new_markdown_cell(text_combine),
    nbf.v4.new_code_cell(code_combine)
]

with open('moo_solver.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook generated successfully.")
