# Grey Wolf Optimization for Feature Selection

## Overview
This project applies the Grey Wolf Optimization (GWO) algorithm for feature selection on the Chronic Kidney Disease dataset. It identifies the most informative features and evaluates the model's performance using a Random Forest classifier. By selecting fewer, more relevant features, the project aims to improve interpretability and maintain high predictive accuracy.

## Files
- `gwo.py`: Contains the implementation of the GWO algorithm, including the initialization of wolf positions and the iterative optimization process.
- `fitness_functions.py`: Defines the fitness function used to evaluate and guide the feature selection process.
- `main.py`: The main script that loads the dataset, runs GWO for feature selection, trains a Random Forest model, and reports performance metrics.
- `requirements.txt`: Lists the dependencies needed to run the project.
- `README.md`: Provides an overview of the project and instructions for running it.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Execute the project:  
   ```bash
   python main.py
   ```
---
