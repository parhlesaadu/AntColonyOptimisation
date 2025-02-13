```
Algorithm 5:  Adaptive Local Search
Args:
        seeds (list): Current seed solutions
        fitness_values (list): Fitness values of seed solutions
        local_std (float): Standard deviation for Gaussian perturbation
        num_samples (int): Number of sampled individuals for local search
        eta (float): Small constant for numerical stability

Returns:
        updated_seeds (list): Updated seed solutions
        updated_fitness_values (list): Updated fitness values
```
```
Algorithm 7: Local Search-Based AMS-ACO (LAMS-ACO)

Args:
        NP (int): Ant colony size (number of solutions)
        G (list): Niching size set
        local_std (float): Standard deviation for local search
        max_iterations (int): Maximum number of iterations
        dimension (int): Dimensionality of the solution space
        eta (float): Small constant for numerical stability
        xi (float): Perturbation scaling factor for Algorithm 4

Returns:
        archive (list): Archive of solutions
        fitness_values (list): Fitness values of the solutions
```
