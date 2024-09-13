import numpy as np
import matplotlib.pyplot as plt
import SA
from test import City, PSO


def HPSOSA(num_trucks, file_name, pso_iterations, pso_population_size, p_best_prob, g_best_prob, initial_temp,
           cooling_rate, max_iterations, truck_capacity):
    city_data_file = f'C:/Users/Jimmy_03/PycharmProjects/Gptest/test_data/{file_name}.data'
    city_data = []
    with open(city_data_file, 'r') as file:
        for line in file:
            x_coord, y_coord = map(int, line.strip().split())
            city_data.append((x_coord, y_coord))
    cities = [City(x, y, 1) for x, y in city_data]
    locations = cities
    depot = 0

    # Particle swarm optimization Algorithm
    pso = run_pso(cities, g_best_prob, num_trucks, p_best_prob, pso_iterations, pso_population_size)

    # Simulated Annealing Algorithm
    truck_routes, best_distance, best_route = run_sa(cities, cooling_rate, initial_temp, max_iterations, truck_capacity,
                                                     num_trucks, pso)

    # Assign routes to trucks
    Assign_print(truck_routes, best_distance)
    plot_final(depot, locations, num_trucks, truck_routes)

    return best_distance


def run_sa(cities, cooling_rate, initial_temp, max_iterations, truck_capacity, num_trucks, pso):
    initial_route = [item for sublist in pso.g_best.p_best for item in sublist]
    sa = SA.SimulatedAnnealing(cities=cities, depot=0, initial_temp=initial_temp, cooling_rate=cooling_rate,
                               max_iterations=max_iterations, truck_capacity=truck_capacity, num_trucks=num_trucks)
    truck_routes, best_route, best_distance = sa.run(initial_route)
    return truck_routes, best_distance, best_route


def run_pso(cities, g_best_prob, num_trucks, p_best_prob, pso_iterations, pso_population_size):
    pso = PSO(iterations=pso_iterations, population_size=pso_population_size, num_salesmen=num_trucks,
              p_best_probability=p_best_prob, g_best_probability=g_best_prob, Cities=cities)
    pso.run()
    print(f'cost: {pso.g_best.p_best_cost}\t| g_best: {pso.g_best.p_best}')
    print(f"---")
    return pso


def Assign_print(truck_routes, best_distance):
    # Output the best distance and the routes for each truck
    print("Best total distance:", best_distance)
    for i, route in enumerate(truck_routes):
        print(f"Truck {i + 1}: {route}")


def plot_final(depot, locations, num_trucks, truck_routes):
    plt.figure(figsize=(12, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, num_trucks))

    for i, route in enumerate(truck_routes):
        # Get coordinates for each location in the route
        route_coordinates = [(locations[location].x, locations[location].y) for location in route]

        # Ensure route_coordinates is a 2D array
        route_coordinates = np.array(route_coordinates)

        # If the route is not empty, plot it
        if route_coordinates.size > 0:
            plt.plot(route_coordinates[:, 0], route_coordinates[:, 1], color=colors[i], marker='o',
                     label=f'Truck {i + 1}')

    # Plot the depot
    depot_coordinates = (locations[depot].x, locations[depot].y)
    plt.scatter(depot_coordinates[0], depot_coordinates[1], color='red', s=100, label='Depot')

    plt.title('Hybrid PSO and SA Final solution')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.show()
    plt.show()


def Simulation(type, num_trucks, truck_capacity):
    fitness_es = []
    if type == 'num_trucks':
        choice = 'number of trucks'
        for i in range(1, num_trucks):
            print(f"Iteration = {i}")
            best_distance = HPSOSA(num_trucks=i, file_name='test_pso', pso_iterations=200, pso_population_size=20,
                                   p_best_prob=0.8,
                                   g_best_prob=0.1, initial_temp=10000, cooling_rate=0.995,
                                   max_iterations=10000, truck_capacity=truck_capacity)
            fitness_es.append([best_distance, i])
    else:
        choice = 'truck capacity'
        for i in range(1, truck_capacity):
            best_distance = HPSOSA(num_trucks=num_trucks, file_name='test_pso', pso_iterations=200,
                                   pso_population_size=20, p_best_prob=0.8,
                                   g_best_prob=0.1,
                                   initial_temp=10000, cooling_rate=0.995, max_iterations=10000, truck_capacity=15)
            fitness_es.append([best_distance, i])
    max_best_distance, affiliated_i = min(fitness_es, key=lambda x: x[0])
    print(f"-----------")
    print(f"The optimal {choice} is: {affiliated_i}")
    print(f"With distance = {max_best_distance}")
    print(fitness_es)


# Run Hybrid PSO and SA
HPSOSA(num_trucks=5, file_name='test_pso', pso_iterations=200, pso_population_size=20, p_best_prob=0.8, g_best_prob=0.1,
       initial_temp=10000, cooling_rate=0.995, max_iterations=10000, truck_capacity=15)

# Simulation('truck_capacity', 5, 30)
