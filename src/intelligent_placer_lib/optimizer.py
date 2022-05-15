import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import pygad


algorithm_param = {'max_num_iteration': 40,\
                   'population_size': 60,\
                   'mutation_probability':0.15,\
                   'elit_ratio': 0.05,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.1,\
                   'crossover_type': 'two_point',\
                   'max_iteration_without_improv':None}


def optimize(loss, bounds, callbacks=None):
    model = ga(function=loss,
               dimension=len(np.array(bounds)),
               variable_type='int',
               variable_boundaries=np.array(bounds),
               algorithm_parameters=algorithm_param)
    model.run()
    return model.output_dict['function']


def optimize_pygad(loss, bounds, callbacks=None):
    num_generations = 100
    num_parents_mating = 2

    sol_per_pop = 50
    num_genes = len(bounds)

    gene_type=int
    init_range_low = -20
    init_range_high = 20

    parent_selection_type = "sss"
    keep_parents = 1

    crossover_type = "single_point"

    mutation_type = "random"
    mutation_percent_genes = 50

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=loss,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           init_range_low=init_range_low,
                           init_range_high=init_range_high,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           gene_type=gene_type,
                           mutation_percent_genes=mutation_percent_genes,
                           random_mutation_min_val=-5.0,
                           random_mutation_max_val=5.0,
                           stop_criteria="reach_200")

    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=loss(solution, 1)))
    return solution_fitness
