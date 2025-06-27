#  SoC_2025_Kungfu_Master

This repository contains two assignments as part of the machine learning project. Each task was implemented from scratch without using pre-built models, to strengthen understanding of the underlying algorithms.

##  Assignment 1: Linear Regression (from scratch)

### Objective
Build a **Simple Linear Regression** model from scratch (without using sklearn’s `LinearRegression`) to predict a target variable using one or more input features.

### Dataset
You are provided with:
- `train.csv` containing both input features and target (`MedHouseVal`)
- `test.csv` containing only input features

### Implementation Details
Implemented a custom `LinearRegression` class with methods:
- Implemented a custom `LinearRegression` class with methods:
  - `__init__()` – Initialize weights, bias, learning rate
  - `fit(X, y)` – Perform training using **gradient descent**
  - `predict(X)` – Predict target using learned weights
- Used `train_test_split`, `pandas`, `numpy`, and `matplotlib` (allowed)
- Visualized model performance with plots
- Generated `submission.csv` with predictions for the test set

### Evaluation Metric
- **Root Mean Squared Error (RMSE)** on predictions
- Example format of the output file:
- row_id,MedHouseVal
- 0,2.01
- 1,0.92
- 2,1.11

  


##  Assignment 2: Multi-Armed Bandit Algorithms

###  Objective
Implement and evaluate the following **bandit algorithms** from scratch:
- Epsilon-Greedy
- Upper Confidence Bound (UCB)
- Thompson Sampling
###  Implementation Details
- Created a base `MultiBandit` environment with multiple arms and changeable reward probabilities.
- Implemented each algorithm as a separate class with:
- `select_arm()` – Arm selection logic
- `run_algorithm()` – Run for a specified horizon
- `give_best_arm()` – Return best estimated arm
- `plot()` – Plot regret over time
- Added a unified `evaluate_algorithms()` function to:
- Accept custom probabilities
- Run all 3 algorithms on the same bandit
- Print total regret and selected best arm
- Plot regret curves for comparison
- Print the **worst-performing algorithm** (highest regret)

### Example Output
Evaluating on custom bandit: [0.1, 0.4, 0.8, 0.3]

- Epsilon-Greedy (e=0.2): Total Regret = 12.00, Best Arm = 2
- UCB : Total Regret = 22.10, Best Arm = 2
- Thompson Sampling: Total Regret = 4.00, Best Arm = 2

Based on the current run, the least effective algorithm is: UCB (Regret = 22.10)

## What I Learned

- How to implement gradient descent manually and understand convergence behavior.
- How Linear Regression operates at a mathematical level.
- Core concepts of **exploration vs exploitation** in bandit algorithms.
- Hands-on with regret analysis and evaluation of online learning strategies.

## Files Included




---

## Author

Srinithya Devraj  
Undergraduate Student, IIT Bombay






