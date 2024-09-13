import random
import numpy as np
import matplotlib.pyplot as plt
from Particle import Particle
from City import City

class PSO:
    # Initialize the PSO with the given parameters
    def __init__(self, iterations, population_size, num_salesmen, g_best_probability=1.0, p_best_probability=1.0,
                 Cities=None):
        self.cities = Cities
        self.num_cities = len(Cities)
        self.num_salesmen = num_salesmen
        self.g_best = None
        self.g_cost_iter = []
        self.iterations = iterations
        self.population_size = population_size
        self.particles = []
        self.g_best_probability = g_best_probability
        self.p_best_probability = p_best_probability
        self.max_no_improvement = 200
        self.no_improvement_count = 0

        solutions = self.initial_population()
        self.particles = \
            [Particle(routes=solution, num_vehicles=num_salesmen, locations=Cities) for solution in solutions]

    def distribute_cities(self):
        # Generate random points and sort them
        random_points = np.sort(np.random.randint(1, self.num_cities, self.num_salesmen - 1))
        # Compute the number of cities for each vehicle by taking the differences between sorted points
        cities_per_vehicle = np.diff(np.concatenate(([0], random_points, [self.num_cities])))

        return cities_per_vehicle

    def greedy_algorithm(self):
        population = []

        for _ in range(self.population_size):
            remaining_cities = list(range(self.num_cities))
            random.shuffle(remaining_cities)  # Shuffle the order of remaining cities to introduce randomness
            vehicle_routes = [[] for _ in range(self.num_salesmen)]

            while remaining_cities:
                for vehicle_index in range(self.num_salesmen):
                    if not remaining_cities:
                        break

                    if not vehicle_routes[vehicle_index]:
                        # If the vehicle has no assigned city, assign a random city
                        current_city_index = remaining_cities.pop(0)
                    else:
                        # Get the last city in the vehicle's route
                        last_city_index = vehicle_routes[vehicle_index][-1]
                        current_city = self.cities[last_city_index]
                        # Find the closest city that is still available
                        current_city_index = min(remaining_cities,
                                                 key=lambda idx: current_city.distance(self.cities[idx]))
                        remaining_cities.remove(current_city_index)

                    vehicle_routes[vehicle_index].append(current_city_index)

            population.append(vehicle_routes)

        return population

    def initial_population(self):
        population = []
        for _ in range(self.population_size):
            city_indices = list(range(self.num_cities))
            random.shuffle(city_indices)
            cities_distribution = self.distribute_cities()

            random_population = []
            start_idx = 0
            for num_cities in cities_distribution:
                end_idx = start_idx + num_cities
                vehicle_cities = city_indices[start_idx:end_idx]
                random_population.append(vehicle_cities)
                start_idx = end_idx
            population.append(random_population)
        return population

    def add_depot(self, particle):
        for j in range(self.num_salesmen):
            if particle[j]:
                if particle[j][0] != 0:
                    particle[j].insert(0, 0)  # Insert 0 at the beginning
                if particle[j][-1] != 0:
                    particle[j].append(0)  # Append 0 at the end
        return particle

    def run(self):
        self.g_best = min(self.particles, key=lambda p: p.p_best_cost)  # Find initial global best
        print(f"Initial best cost = {self.g_best}")
        best_cost = self.g_best.p_best_cost

        for itr in range(self.iterations):
            self.g_best = min(self.particles, key=lambda p: p.p_best_cost)  # Update global best
            if self.g_best.p_best_cost < best_cost:
                best_cost = self.g_best.p_best_cost
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1

            if self.no_improvement_count >= self.max_no_improvement:
                print(f"Stopping early at iteration {itr} due to no improvement.")
                break

            for particle in self.particles:
                particle.clear_velocity()

                converted_p_best = convert_array(particle.p_best, self.num_cities, self.num_salesmen)

                temp_velocity = []

                g_best = [route[:] for route in self.g_best.p_best]
                new_routes = [route[:] for route in particle.routes]

                g_best_converted = convert_array(g_best, self.num_cities, self.num_salesmen)
                new_routes_converted = convert_array(new_routes, self.num_cities, self.num_salesmen)

                for i in range(self.num_salesmen):
                    for j in range(self.num_cities):
                        if new_routes_converted[i][j] != converted_p_best[i][j] and new_routes_converted[i][j] != -1:
                            a, b = find_index(new_routes_converted[i][j], converted_p_best, self.num_salesmen,
                                              self.num_cities)
                            swap = (i, j, a, b, self.p_best_probability)
                            temp_velocity.append(swap)
                            new_routes_converted[swap[0]][swap[1]], new_routes_converted[swap[2]][swap[3]] = \
                                new_routes_converted[swap[2]][swap[3]], new_routes_converted[swap[0]][swap[1]]

                for i in range(self.num_salesmen):
                    for j in range(self.num_cities):
                        if new_routes_converted[i][j] != g_best_converted[i][j] and new_routes_converted[i][j] != -1:
                            a, b = find_index(new_routes_converted[i][j], g_best_converted, self.num_salesmen,
                                              self.num_cities)
                            swap = (i, j, a, b, self.g_best_probability)
                            temp_velocity.append(swap)
                            g_best_converted[swap[0]][swap[1]], g_best_converted[swap[2]][swap[3]] = \
                                g_best_converted[swap[2]][swap[3]], g_best_converted[swap[0]][swap[1]]

                particle.velocity = temp_velocity
                for swap in temp_velocity:
                    if random.random() <= swap[4]:
                        new_routes_converted[swap[0]][swap[1]], new_routes_converted[swap[2]][swap[3]] = \
                            new_routes_converted[swap[2]][swap[3]], new_routes_converted[swap[0]][swap[1]]

                new_routes = revert_array(new_routes_converted, self.num_salesmen)
                for r in range(self.num_salesmen):
                    particle.routes[r] = new_routes[r]

                particle.update_costs_and_p_best()
            # self.plot_routes(itr)
        # self.add_depot(self.g_best)

    def plot_routes(self, itr):
        if itr % 20 == 0:
            plt.figure(0)
            plt.plot(self.g_cost_iter, 'g')
            plt.ylabel('Distance')
            plt.xlabel('Generation')

            figure = plt.figure(0)
            figure.suptitle('pso iter')

            figure = plt.figure(1)
            figure.clear()
            figure.suptitle(f'pso TSP iter {itr}')

            for route in self.g_best.p_best:
                x_list, y_list = [], []
                for city_index in route:
                    city = cities[city_index]
                    x_list.append(city.x)
                    y_list.append(city.y)
                x_list.append(cities[route[0]].x)
                y_list.append(cities[route[0]].y)
                plt.plot(x_list, y_list, marker='o')

            plt.draw()
            plt.pause(.001)
        self.g_cost_iter.append(self.g_best.p_best_cost)

# Helper Functions:
def convert_array(particle, num_cities, num_salesmen):
    converted_particle = np.full([num_salesmen, num_cities], -1)
    for i in range(num_salesmen):
        converted_particle[i][:len(particle[i])] = particle[i]
    return converted_particle

def revert_array(particle, num_salesmen):
    reverted_particle = []
    for i in range(num_salesmen):
        indices_to_remove = np.where(particle[i] == -1)
        reverted_particle.append(np.delete(particle[i], indices_to_remove).tolist())
    return reverted_particle

def find_index(number, particle, num_vehicles, num_cities):
    # find index of place where elements will be swapped between particle and best particle
    for i in range(num_vehicles):
        for j in range(num_cities):
            if particle[i][j] == number:
                return i, j

def plot_etc():
    fig = plt.figure(1)
    fig.suptitle('pso TSP')
    for route in pso.g_best.p_best:
        x_list, y_list = [], []
        for city_index in route:
            city = cities[city_index]
            x_list.append(city.x)
            y_list.append(city.y)
        x_list.append(cities[route[0]].x)
        y_list.append(cities[route[0]].y)
        plt.plot(x_list, y_list, marker='o')
    plt.show(block=True)


if __name__ == "__main__":
    # Read city data from file
    city_data_file = '../test_data/test_pso.data'
    city_data = []

    with open(city_data_file, 'r') as file:
        for line in file:
            x_coord, y_coord = map(int, line.strip().split())
            city_data.append((x_coord, y_coord))

    cities = [City(x, y, 1) for x, y in city_data]

    pso = PSO(iterations=300, population_size=20, num_salesmen=5, p_best_probability=0.3, g_best_probability=0.5,
              Cities=cities)
    pso.run()
    print(f'cost: {pso.g_best.p_best_cost}\t| g_best: {pso.g_best.p_best}')
    plot_etc()
