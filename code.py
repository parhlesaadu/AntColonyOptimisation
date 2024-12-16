import numpy as np

def fitness_function(x):
    return np.sin(5 * np.pi * x) ** 6

# Algorithm 1: Clustering for Crowding
def algorithm1(archive, cluster_size):
    archive_array = np.array(archive)
    clusters = []

    while len(archive_array) > 0:
        reference_point = np.random.uniform(0, 10, archive_array.shape[1])
        distances = np.linalg.norm(archive_array - reference_point, axis=1)
        cluster_indices = np.argsort(distances)[:cluster_size]
        cluster = archive_array[cluster_indices]
        clusters.append(cluster)
        archive_array = np.delete(archive_array, cluster_indices, axis=0)

    return clusters

# Algorithm 2: Clustering for Speciation
def algorithm2(archive, fitness_values, NS):
    sorted_indices = np.argsort(fitness_values)
    sorted_archive = np.array(archive)[sorted_indices]
    species = []
    while len(sorted_archive) > 0:
        best = sorted_archive[0]
        distances = np.linalg.norm(sorted_archive - best, axis=1)
        nearest_indices = np.argsort(distances)[:NS]
        species.append(sorted_archive[nearest_indices])
        sorted_archive = np.delete(sorted_archive, nearest_indices, axis=0)
    return species


# Algorithm 4: Solution Construction
def algorithm4(niches, fitness, FSmax, FSmin, eta, archive, xi, NP):
    results = []
    for niche in niches:
        niche_indices = [np.where((archive == solution).all(axis=1))[0][0] for solution in niche]
        niche_fitness = fitness[niche_indices]
        niche_archive = archive[niche_indices]

        FSi_max = np.max(niche_fitness)
        FSi_min = np.min(niche_fitness)
        sigma_i = 0.1 + 0.3 * np.exp(-(FSi_max - FSi_min) / (FSmax - FSmin + eta))

        weights = np.exp(-((np.arange(len(niche)) - 1) ** 2) / (2 * sigma_i ** 2 * len(niche) ** 2))
        probabilities = weights / np.sum(weights)
        cumulative_probabilities = np.cumsum(probabilities)

        random_value = np.random.rand()
        selected_index = np.searchsorted(cumulative_probabilities, random_value)
        selected_solution = niche_archive[selected_index]

        delta = xi * np.mean(np.abs(niche_fitness - niche_fitness[selected_index]))
        for _ in range(NP):
            solution = selected_solution + np.random.normal(0, delta, size=selected_solution.shape)
            results.append(solution)

    return results

# Algorithm 5: Adaptive Local Search
def algorithm5(seeds, fitness_values, local_std, num_samples, eta=0.01):
    def evaluate_fitness(solution):
        return sum(np.sin(5 * np.pi * x) ** 6 if 0 <= x <= 1 else -np.inf for x in solution)

    num_seeds = len(seeds)
    FSEmin = np.min(fitness_values)
    FSEmax = np.max(fitness_values)

    # Calculate selection probabilities for each seed
    probabilities = [
        (fitness - FSEmin) / (FSEmax - FSEmin + eta) for fitness in fitness_values
    ]

    updated_seeds = seeds.copy()
    updated_fitness_values = fitness_values.copy()

    for i in range(num_seeds):
        if np.random.rand() <= probabilities[i]:  # Check if seed is selected for improvement
            for _ in range(num_samples):
                # Generate a new individual using Gaussian perturbation
                new_individual = np.clip(
                    seeds[i] + np.random.normal(0, local_std, size=len(seeds[i])),
                    0,
                    1,
                )  # Ensure solution stays within [0, 1]

                new_fitness = evaluate_fitness(new_individual)
                if new_fitness > updated_fitness_values[i]:  # Update if better
                    updated_seeds[i] = new_individual
                    updated_fitness_values[i] = new_fitness

    return updated_seeds, updated_fitness_values

# Algorithm 6: LAMC-ACO
def algorithm6(NP, G, delta, termination_criterion, eta, xi):
    # Initialize the archive with random solutions
    archive = np.array([np.random.uniform(0, 10, size=5) for _ in range(NP)])
    # Calculate the fitness for each solution in the archive
    fitness_archive = []
    for sol in archive:
        for x in sol:
            fitness_archive.append(fitness_function(x))  # Adjusted for individual scalar values
    fitness_archive = np.array(fitness_archive)

    iteration = 0
    while iteration < termination_criterion:
        FSmax = np.max(fitness_archive)
        FSmin = np.min(fitness_archive)
        NS = np.random.choice(G)
        niches = algorithm1(archive, NS)

        new_solutions = algorithm4(niches, fitness_archive, FSmax, FSmin, eta, archive, xi, NS)

        for c_k in new_solutions:
            for x in c_k:  # Iterate over each scalar element in c_k
                nearest_idx = np.argmin([euclidean_distance(x, sol) for sol in archive])
                if fitness_function(x) > fitness_archive[nearest_idx]:
                    archive[nearest_idx] = x
                    fitness_archive[nearest_idx] = fitness_function(x)

        archive, fitness_archive = algorithm5(archive, fitness_archive, delta, num_samples=1)
        iteration += 1
    
    return archive, fitness_archive

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

# Algorithm 7: LAMS-ACO
def algorithm7(NP, G, local_std, max_iterations, dimension, eta=0.01, xi=0.1):
    def evaluate_fitness(solution):
        return sum(np.sin(5 * np.pi * x) ** 6 if 0 <= x <= 1 else -np.inf for x in solution)

    # Step 1: Initialize NP solutions
    archive = np.random.uniform(0, 1, (NP, dimension))
    fitness_values = np.array([evaluate_fitness(ind) for ind in archive])

    for iteration in range(max_iterations):
        # Step 2: Obtain FSmax and FSmin
        FSmax, FSmin = np.max(fitness_values), np.min(fitness_values)

        # Step 3: Randomly select a niching size NS from G
        NS = np.random.choice(G)

        # Step 4: Partition the archive into species using Algorithm 2
        species = algorithm2(np.array(archive), fitness_values, NS)

        # Step 5: Construct NP solutions using Algorithm 4
        archive_array = np.array(archive)
        new_solutions = algorithm4(species, fitness_values, FSmax, FSmin, eta, archive_array, xi, NP)

        # Step 6: Update species with new solutions
        updated_species = []
        for niche in species:
            niche_array = np.array(niche)
            for new_solution in new_solutions:
                distances = np.linalg.norm(niche_array - new_solution, axis=1)
                nearest_index = np.argmin(distances)
                if evaluate_fitness(new_solution) > evaluate_fitness(niche_array[nearest_index]):
                    niche_array[nearest_index] = new_solution
            updated_species.append(niche_array)

        # Flatten updated species back into the archive
        archive = np.vstack(updated_species)

        # Step 7: Perform local search using Algorithm 5
        archive, fitness_values = algorithm5(archive, fitness_values, local_std, num_samples=5, eta=eta)

        # Step 8: Termination criterion
        if iteration >= max_iterations:
            break

    return archive, fitness_values

# Parameters for algorithm 6
NP = 10
G = [2, 3, 4]
delta = 1.0
termination_criterion = 50
eta = 1.0
xi = 0.5
final_archive, final_fitness = algorithm6(NP, G, delta, termination_criterion, eta, xi)
print("Final Archive (Crwod-based clustering):")
for sol, fit in zip(final_archive, final_fitness):
        print(f"Solution: {sol}, Fitness: {fit}")

# Parameters for Algorithm 7
NP = 10  # Ant colony size
G = [2, 3, 4]  # Niching size set
local_std = 0.1  # Local search standard deviation
termination_criterion = 50  # Number of iterations
dimension = 5  # Dimensionality of solutions
eta = 0.01  # Small constant for numerical stability
xi = 0.5  # Perturbation scaling factor
final_archive, final_fitness = algorithm7(NP, G, local_std, termination_criterion, dimension, eta, xi)
print("Final Archive (Species-based clustering):")
for sol, fit in zip(final_archive, final_fitness):
    print(f"Solution: {sol}, Fitness: {fit:.4f}")
