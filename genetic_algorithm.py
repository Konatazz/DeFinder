import numpy as np
import torch
import random
from torchvision import transforms
from get_coverage import coverage

def initialize_population(image, mask, pop_size):
    """
    """
    population = []
    for _ in range(pop_size):
        noise = np.random.uniform(-0.1, 0.1, image.shape) * mask
        new_image = image + noise
        population.append(new_image)
    return population

def fitness_function(model, population):
    """
    """
    fitness_scores = []
    # nc_evaluator = get_coverage.coverage.nc(model, threshold=0.6)
    for image in population:
        image_tensor = transforms.ToTensor()(image).unsqueeze(0).float().to('cuda:0')
        preds = model(image_tensor)
        error = -preds.max(1)[0]
        nc_coverage = coverage.calculate_nc(model, image_tensor.squeeze(0), threshold=0.6)
        nc_score = np.mean(list(nc_coverage.values()))
        fitness_score = (w1 * error.item()) + (w2 * nc_score)
        fitness_scores.append(fitness_score)
        # fitness_scores.append(error.item())
    return fitness_scores

def select_parents(population, fitness, num_parents):
    """
    """
    parents_idx = np.argsort(fitness)[-num_parents:]
    return [population[i] for i in parents_idx]

def crossover(parents, offspring_size, mask):
    """
    """
    mask = mask.astype(np.float32)
    offspring = []
    for _ in range(offspring_size):
        parent1, parent2 = random.sample(parents, 2)
        child = parent1 * mask + parent2 * (1 - mask)
        offspring.append(child)
    return offspring

def mutate(offspring, mask):
    """
    """
    mask = mask.astype(np.float32)
    for child in offspring:
        mutation = np.random.uniform(-0.1, 0.1, child.shape) * mask
        child += mutation
    return offspring

def genetic_algorithm(image, mask, model, pop_size=50, num_generations=10):
    """
    """
    model = model.to('cuda:0')
    population = initialize_population(image, mask, pop_size)
    for _ in range(num_generations):
        fitness = fitness_function(model, population)
        parents = select_parents(population, fitness, pop_size // 2)
        offspring = crossover(parents, pop_size - len(parents), mask)
        offspring = mutate(offspring, mask)
        population = parents + offspring

        # random_individuals = initialize_population(image, mask, pop_size // 10)
        # population = parents + offspring + random_individuals
    return population[np.argmax(fitness)]
