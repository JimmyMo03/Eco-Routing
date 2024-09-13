import random
import math

class SimulatedAnnealing:
    def __init__(self, cities, depot, initial_temp, cooling_rate, max_iterations, truck_capacity, num_trucks):
        self.cities = cities
        self.depot = depot
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.truck_capacity = truck_capacity
        self.num_trucks = num_trucks

    def distance(self, p1, p2):
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def total_distance(self, path):
        total = 0
        total += self.distance(self.cities[self.depot], self.cities[path[0]])  # From depot to first location
        for k in range(len(path) - 1):
            total += self.distance(self.cities[path[k]], self.cities[path[k + 1]])
        total += self.distance(self.cities[path[-1]], self.cities[self.depot])  # From last location back to depot
        return total

    def get_neighbour(self, path):
        a, b = random.sample(range(len(path)), 2)
        neighbour = path[:]
        neighbour[a], neighbour[b] = neighbour[b], neighbour[a]
        return neighbour

    def assign_routes_to_trucks(self, path):
        demands = {loc: self.cities[loc].demand for loc in path}
        truck_routes = [[] for _ in range(self.num_trucks)]
        truck_loads = [0] * self.num_trucks

        for location in path:
            for i in range(self.num_trucks):
                if truck_loads[i] + demands[location] <= self.truck_capacity:
                    truck_routes[i].append(location)
                    truck_loads[i] += demands[location]
                    break

        for route in truck_routes:
            route.insert(0, self.depot)
            route.append(self.depot)

        return truck_routes

    def run(self, init_route):
        current_route = init_route
        current_distance = self.total_distance(current_route)
        temperature = self.initial_temp

        optimal_route = current_route
        optimal_distance = current_distance

        for _ in range(self.max_iterations):
            neighbour = self.get_neighbour(current_route)
            neighbour_distance = self.total_distance(neighbour)

            if neighbour_distance < current_distance or random.random() < math.exp(
                    (current_distance - neighbour_distance) / temperature):
                current_route = neighbour
                current_distance = neighbour_distance

                if current_distance < optimal_distance:
                    optimal_route = current_route
                    optimal_distance = current_distance

            temperature *= self.cooling_rate

            if temperature < 1e-10:
                break

        truck_routes = self.assign_routes_to_trucks(optimal_route)
        return truck_routes, optimal_route, optimal_distance
