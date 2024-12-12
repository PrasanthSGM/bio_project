import numpy as np

class GreyWolfOptimizer:
    def __init__(self, fitness_function, dim, pop_size=10, max_iter=50, lb=0, ub=1):
        """
        Grey Wolf Optimizer for feature selection
        
        Args:
        fitness_function (callable): Function to evaluate wolf positions
        dim (int): Dimension of the search space
        pop_size (int): Population size
        max_iter (int): Maximum number of iterations
        lb (float): Lower bound of search space
        ub (float): Upper bound of search space
        """
        self.fitness_function = fitness_function
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub
        
        # Wolf hierarchy
        self.alpha_pos = None
        self.beta_pos = None
        self.delta_pos = None
        
        # Wolf scores
        self.alpha_score = float("inf")
        self.beta_score = float("inf")
        self.delta_score = float("inf")
        
        # Initialize population
        self.population = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        
        # Track fitness history for convergence analysis
        self.fitness_history = []

    def optimize(self):
        """
        Perform Grey Wolf Optimization
        
        Returns:
        tuple: Best solution position and its fitness score
        """
        # Initial population evaluation
        for i in range(self.pop_size):
            fitness = self.fitness_function(self.population[i, :])
            
            # Update wolf hierarchy for the first iteration
            if self.alpha_pos is None:
                self.alpha_score = fitness
                self.alpha_pos = self.population[i, :].copy()
            elif fitness < self.alpha_score:
                # Demote existing wolves
                self.delta_score = self.beta_score
                self.delta_pos = self.beta_pos.copy() if self.beta_pos is not None else self.population[i, :].copy()
                self.beta_score = self.alpha_score
                self.beta_pos = self.alpha_pos.copy()
                
                # Promote new wolf
                self.alpha_score = fitness
                self.alpha_pos = self.population[i, :].copy()
            
            elif self.beta_pos is None or fitness < self.beta_score:
                # Demote existing delta wolf if needed
                self.delta_score = self.beta_score
                self.delta_pos = self.beta_pos.copy() if self.beta_pos is not None else self.population[i, :].copy()
                
                # Promote new wolf
                self.beta_score = fitness
                self.beta_pos = self.population[i, :].copy()
            
            elif self.delta_pos is None or fitness < self.delta_score:
                self.delta_score = fitness
                self.delta_pos = self.population[i, :].copy()

        # Main optimization loop
        for iteration in range(self.max_iter):
            # Decay coefficient
            a = 2 - 2 * iteration / self.max_iter

            # Update wolf positions
            for i in range(self.pop_size):
                # Random coefficients
                r1, r2 = np.random.random(), np.random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                
                # Distance and position calculations
                D_alpha = abs(C1 * self.alpha_pos - self.population[i, :])
                X1 = self.alpha_pos - A1 * D_alpha

                r1, r2 = np.random.random(), np.random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * self.beta_pos - self.population[i, :])
                X2 = self.beta_pos - A2 * D_beta

                r1, r2 = np.random.random(), np.random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * self.delta_pos - self.population[i, :])
                X3 = self.delta_pos - A3 * D_delta

                # Update position
                self.population[i, :] = (X1 + X2 + X3) / 3
                
                # Ensure position is within bounds
                self.population[i, :] = np.clip(self.population[i, :], self.lb, self.ub)

                # Re-evaluate fitness after position update
                fitness = self.fitness_function(self.population[i, :])
                
                # Update wolf hierarchy based on new fitness
                if fitness < self.alpha_score:
                    # Demote existing wolves
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()
                    self.beta_score = self.alpha_score
                    self.beta_pos = self.alpha_pos.copy()
                    
                    # Promote new wolf
                    self.alpha_score = fitness
                    self.alpha_pos = self.population[i, :].copy()
                
                elif fitness < self.beta_score:
                    # Demote existing delta wolf
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()
                    
                    # Promote new wolf
                    self.beta_score = fitness
                    self.beta_pos = self.population[i, :].copy()
                
                elif fitness < self.delta_score:
                    self.delta_score = fitness
                    self.delta_pos = self.population[i, :].copy()
            
            # Store fitness for convergence tracking
            self.fitness_history.append(self.alpha_score)

        return self.alpha_pos, self.alpha_score