"""Simple Evolutionary Algorithm (EA) as implemented by DEAP."""

import matplotlib.pyplot as plt
import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

import functions.numpy_functions as npf

dims = 200
function = npf.Sphere()


def eval_function(individual):
  return function(np.array(individual, dtype=np.float32)),


creator.create("FitnessMinimum", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMinimum)

toolbox = base.Toolbox()

toolbox.register("attr_position", np.random.uniform, function.domain.min,
                 function.domain.max)

toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_position, dims)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", eval_function)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
  population = toolbox.population(n=500)
  hall_of_fame = tools.HallOfFame(1)
  stats = tools.Statistics(lambda ind: ind.fitness.values)
  stats.register("avg", np.mean)
  stats.register("best", np.min)

  individuals, log_book = algorithms.eaSimple(population, toolbox,
                                              cxpb=0.5, mutpb=0.2, ngen=2000,
                                              stats=stats,
                                              halloffame=hall_of_fame,
                                              verbose=True)

  return individuals, log_book, hall_of_fame


def evaluate_ea(logs: tools.Logbook, hall_of_fame: tools.HallOfFame, func, d):
  _, ax = plt.subplots(figsize=(25.0, 8.0,))

  objective_values = logs.select('best')

  y_min = np.min(objective_values)
  ax.plot(range(len(objective_values)), objective_values,
          label='Best value found: {:.2f}'.format(y_min))
  ax.set(xlabel="Generation\nBest solution at: {0}".format(hall_of_fame[0]),
         ylabel="Best objective value",
         title="eaSimple on {0} ({1} Dims)".format(func.name, d))

  x_ticks = np.arange(0, len(logs), step=50.0)
  x_labels = ['{:.0f}'.format(val) for val in x_ticks]

  ax.set_xticks(x_ticks)
  ax.set_xticklabels(x_labels)
  ax.set_xscale('symlog', base=2)
  ax.set_xlim(left=0)

  ax.legend()
  ax.grid()

  plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
           rotation_mode="anchor")
  plt.savefig(fname='{0}-{1}dims.png'.format(func.name, d), bbox_inches='tight')


if __name__ == "__main__":
  pop, log, hof = main()
  evaluate_ea(log, hof, function, dims)
