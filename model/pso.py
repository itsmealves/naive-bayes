import numpy as np

class ParticleSwarmOptimization(object):
    class __Particle(object):
        def __init__(self, dimensions):
            mean = 0
            std_dev = 5.0
            self.__pbest = (np.zeros(dimensions), 0)
            self.__speed = np.random.normal(mean, std_dev, dimensions)
            self.__position = np.random.normal(mean, std_dev, dimensions)

        def current_position(self):
            return self.__position

        def best_position(self):
            return self.__pbest

        def evaluate(self, fn):
            v = fn(self.__position)

            if v > self.__pbest[1]:
                self.__pbest = (self.__position, v)

        def update(self, gbest, i, c1, c2):
            r1 = np.random.random()
            r2 = np.random.random()

            inertia = self.__speed * i
            social = c2 * r2 * (gbest[0] - self.__position)
            cognitive = c1 * r1 * (self.__pbest[0] - self.__position)

            self.__speed = inertia + cognitive + social
            self.__position = self.__position + self.__speed

    def __init__(self, dimensions=None, swarm_size=50, cognitive=2.5, social=1.5):
        self.__social = social
        self.__cognitive = cognitive
        self.__swarm_size = swarm_size
        self.__dimensions = dimensions

    def __initialize_swarm(self):
        spawner = lambda x: self.__Particle(self.__dimensions)
        return map(spawner, range(self.__swarm_size))

    def __get_swarm_best_position(self, swarm):
        get_value = lambda x: x[1]
        positions = map(lambda x: x.best_position(), swarm)
        sorted_positions = sorted(positions, reverse=True, key=get_value)
        return sorted_positions[0]

    def optimize(self, objective_function, iterations=200):
        it = 0
        inertia = 0.9
        inertia_step = 0.5 / iterations
        swarm = self.__initialize_swarm()
        gbest = (np.zeros(self.__dimensions), 0)

        while it < iterations:
            it = it + 1
            for particle in swarm:
                particle.evaluate(objective_function)
                particle.update(gbest, inertia, self.__cognitive, self.__social)

            inertia = inertia - inertia_step
            gbest = self.__get_swarm_best_position(swarm)

        return gbest

