from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
import numpy as np


class GeneticSelection:

    def __init__(self, n_genomes=100, crossover_rate=0.75, mutation_rate=0.0001, max_generations=100,
                 estimator=RandomForestClassifier(n_jobs=-1), n_features=0, kfold=5):
        self.n_genomes = n_genomes
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.estimator = estimator
        self.kfold = kfold
        self.n_features = n_features
        self.X = None
        self.y = None
        self.genomes = []
        self.scores = []
        self.best_genome = None
        self.best_score = None

    def __fenerate_genome(self):
        return np.random.choice(self.X.columns, self.n_features, replace=False)

    def __initialize_population(self):
        self.genomes = [self.__fenerate_genome() for i in range(self.n_genomes)]
        self.scores = [self.__fitness_function(genome) for genome in self.genomes]
        self.best = self.genomes[np.argmax(self.scores)]
        self.best_score = np.max(self.scores)

    def __fitness_function(self, genome):
        return cross_validate(self.estimator, self.X[genome], self.y, cv=self.kfold)['test_score'].mean()

    def __crossover(self, g1, g2):
        new_g1 = new_g2 = []
        for c1, c2 in zip(g1, g2):
            if np.random.random() <= self.crossover_rate:
                new_g1.append(c1)
                new_g2.append(c2)
            else:
                new_g1.append(c2)
                new_g2.append(c1)
        return self.__mutate(new_g1), self.__mutate(new_g2)

    def __mutate(self, genome):
        if np.random.random() <= self.mutation_rate:
            genome[np.random.randint(self.n_features)] = np.random.choice(self.X.columns)
        return genome

    def __generation(self):
        new_genomes = []
        while len(new_genomes) < self.n_genomes:
            g1, g2 = np.random.choice(a=np.arange(self.n_genomes), size=2,
                                      p=(np.array(self.scores) / np.array(self.scores).sum()))
            g1 = self.genomes[g1]
            g2 = self.genomes[g2]
            g1, g2 = self.__crossover(g1, g2)
            new_genomes += [g1, g2]
        self.genomes = new_genomes
        self.scores = [self.__fitness_function(genome) for genome in self.genomes]
        if np.max(self.scores) >= self.best_score:
            self.best = self.genomes[np.argmax(self.scores)]
            self.best_score = np.max(self.scores)

    def run(self, X, y):
        self.X = X
        self.y = y
        if self.n_features == 0:
            self.n_features = int(len(X) / 2)

        self.__initialize_population()

        for iteration in range(self.max_generations):
            print('Generation {} Best score: {} Mean score: {}'.format(iteration, np.max(self.scores),
                                                                       np.mean(self.scores)))
            if np.max(self.scores) == 1:
                break
            self.__generation()
        return self.best