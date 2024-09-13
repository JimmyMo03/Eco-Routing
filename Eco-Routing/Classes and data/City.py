# import geopy.distance

class City:
    # Represents a city with x and y coordinates
    def __init__(self, x, y, demand):
        self.x = x
        self.y = y
        self.demand = demand

    # Calculate Euclidean distance to another city
    def distance(self, other):
        # c1 = (self.x, self.y)
        # c2 = (other.x, other.y)
        # return geopy.distance.geodesic(c1, c2).km
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def capacity(self):
        return self.demand

    # Equality check for city objects
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    # String representation of a city object
    def __repr__(self):
        return f"City({self.x}, {self.y})"

    def __getitem__(self, key):
        if key == 0:
            return self.x
        else:
            return self.y
