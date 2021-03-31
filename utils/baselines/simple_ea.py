from functions.numpy_functions import *
from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import matplotlib.pyplot as plt

dims = 20
function = RotatedHyperEllipsoid()


def evalFunction(individual):
    return function(np.array(individual, dtype=np.float32)),


creator.create("FitnessMinimum", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMinimum)

toolbox = base.Toolbox()

toolbox.register("attr_position", np.random.uniform, function.domain.min, function.domain.max)

toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_position, dims)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evalFunction)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("best", np.min)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=2000,
                                   stats=stats, halloffame=hof, verbose=True)

    return pop, log, hof


def evaluate_GA(logs: tools.Logbook, hof: tools.HallOfFame, func, d):
    fig, ax = plt.subplots(figsize=(25.0, 8.0,))

    objective_values = logs.select('best')

    y_min = np.min(objective_values)
    ax.plot(range(len(objective_values)), objective_values, label='Best value found: {:.2f}'.format(y_min))
    ax.set(xlabel="Generation\nBest solution at: {0}".format(hof[0]),
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

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.savefig(fname='{0}-{1}dims.png'.format(func.name, d), bbox_inches='tight')


if __name__ == "__main__":
    pop, log, hof = main()
    evaluate_GA(log, hof, function, dims)
