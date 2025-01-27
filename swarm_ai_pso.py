import numpy as np
import random
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class Particle:
    def __init__(self, bounds):
        self.position = np.array([random.uniform(bounds[i][0], bounds[i][1]) for i in range(len(bounds))])
        self.velocity = np.array([random.uniform(-1, 1) for _ in range(len(bounds))])
        self.best_position = self.position.copy()
        self.best_value = float('-inf')

def fitness_function(params):
    C, gamma = params[0], params[1]
    model = SVC(C=C, gamma=gamma, kernel='rbf')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

class PSO:
    def __init__(self, num_particles, bounds, w=0.5, c1=1.5, c2=1.5, max_iter=50):
        self.num_particles = num_particles
        self.bounds = bounds
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive coefficient
        self.c2 = c2  # Social coefficient
        self.max_iter = max_iter
        
        self.swarm = [Particle(bounds) for _ in range(num_particles)]
        self.global_best_position = None
        self.global_best_value = float('-inf')

    def optimize(self):
        for iteration in range(self.max_iter):
            for particle in self.swarm:
                fitness = fitness_function(particle.position)
                
                # Update personal best
                if fitness > particle.best_value:
                    particle.best_value = fitness
                    particle.best_position = particle.position.copy()
                
                # Update global best
                if fitness > self.global_best_value:
                    self.global_best_value = fitness
                    self.global_best_position = particle.position.copy()
                
            for particle in self.swarm:
                r1, r2 = random.random(), random.random()
                cognitive = self.c1 * r1 * (particle.best_position - particle.position)
                social = self.c2 * r2 * (self.global_best_position - particle.position)
                particle.velocity = self.w * particle.velocity + cognitive + social
                
                # Update position and ensure it stays within bounds
                particle.position += particle.velocity
                for i in range(len(self.bounds)):
                    particle.position[i] = np.clip(particle.position[i], self.bounds[i][0], self.bounds[i][1])
            
            print(f"Iteration {iteration+1}: Best Accuracy = {self.global_best_value}")
        
        return self.global_best_position, self.global_best_value

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define search space bounds for C and gamma
bounds = [(0.1, 100), (0.0001, 1)]  # C and gamma ranges

# Run PSO to optimize SVM hyperparameters
pso = PSO(num_particles=20, bounds=bounds, max_iter=30)
best_params, best_accuracy = pso.optimize()

print(f"Optimal Hyperparameters Found: C = {best_params[0]}, Gamma = {best_params[1]}")
print(f"Best Model Accuracy: {best_accuracy}")
