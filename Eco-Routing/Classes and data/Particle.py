class Particle:
    def __init__(self, routes, num_vehicles, locations):
        self.cities = locations
        self.routes = routes
        self.num_vehicles = num_vehicles
        self.p_best = routes
        self.fitness = self.route_cost()
        self.p_best_cost = self.fitness
        self.velocity = []
        self.vehicle_capacity = 15
        self.temp = True

    # Update the costs and personal best if the current cost is better
    def update_costs_and_p_best(self):
        self.fitness = self.route_cost()
        # self.temp = self.cap_cost()
        # print(f"({self.fitness}, {self.p_best_cost}, {self.temp})")
        if self.fitness < self.p_best_cost and self.temp:
            self.p_best = self.routes
            self.p_best_cost = self.fitness

    def route_cost(self):
        # Assuming route is a list of indices
        fit = 0
        for j in range(self.num_vehicles):
            route_len = len(self.routes[j])
            fit += sum(self.distance(self.routes[j][i], self.routes[j][(i + 1) % route_len]) for i in range(route_len))
        return fit

    def cap_cost(self):
        for j in range(self.num_vehicles):
            cap = 0
            route_len = len(self.routes[j])
            cap += sum(self.capacity(self.routes[j][i]) for i in range(route_len))
            if cap > self.vehicle_capacity:
                return False
        return True

    def clear_velocity(self):
        self.velocity.clear()

    def distance(self, city1_index, city2_index):
        city1 = self.cities[city1_index]
        city2 = self.cities[city2_index]
        return city1.distance(city2)

    def capacity(self, city1_index):
        # Assuming cities are represented by indices
        city1 = self.cities[city1_index]
        return city1.capacity()

    def __repr__(self):
        return f"Fitness: {self.fitness}" \
               f"Route: {self.routes}"

    def __len__(self):
        return len(self.routes)

    def __getitem__(self, item):
        return self.routes[item]
