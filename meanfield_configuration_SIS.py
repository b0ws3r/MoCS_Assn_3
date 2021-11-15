import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def graph_exponential_dist():
    z = [int(np.random.exponential(scale=6)) for i in range(1000)]
    if sum(z)%2 != 0:
        z[0] += 1
    configuration_model = nx.configuration_model(z)  # Get_Configuration_Model(z)
    return configuration_model


configuration_model = graph_exponential_dist() # part a & b
degree_freq = nx.degree_histogram(configuration_model)
degrees = range(len(degree_freq))

# Parameters of the model
beta = 0.03 # transmission rate
alpha = .10  # recovery rate

# Initial conditions
I0 = 0.01  # initial fraction of infected nodes
S0 = 1.0 - I0
G_deg_sum = [a * b for a, b in zip(degree_freq, range(0, len(degree_freq)))]
total_degree = sum(G_deg_sum)
avg_k = sum(G_deg_sum) / configuration_model.number_of_nodes()
print(avg_k)
Sk = np.zeros((len(degree_freq)))  # array for expected S_k
Ik = np.zeros((len(degree_freq)))  # array for expected I_k

for k in range(len(degree_freq)):
    # set an expectation of I0 fraction of nodes of degree k infectious
    Sk[k] = degree_freq[k] * S0
    Ik[k] = degree_freq[k] * I0
# Run the model
# Discrete steps of Euler's methods
res = []  # list of results
history = []
S = S0
I = I0  # set initial conditions
h = .1  # timestep
T = np.arange(1, 500 / h)
for t in T:

    # Calculate the mean-field
    theta = 0.0
    for k in range(len(degree_freq)):
        theta += k * Ik[k] / total_degree
    history.append(theta)

    # Set initial global quantities
    S = 0.0
    I = 0.0

    # Run Euler's method for all degree classes k
    for k in range(len(degree_freq)):
        # calculate speeds
        delta_Sk = -beta * k * theta * Sk[k] + alpha * Ik[k]
        delta_Ik = beta * k * theta * Sk[k] - alpha * Ik[k]
        # update dynamical variables
        Sk[k] += delta_Sk * h  # Ik(t+h)
        Ik[k] += delta_Ik * h  # Ik(t+h)
        # update global quantities
        S += Sk[k]
        I += Ik[k]
    res.append((S / configuration_model.number_of_nodes(), I / configuration_model.number_of_nodes()))

# zip unpacked list of tuples (n-th elements all together)
# map them to arrays
St, It = map(np.array, zip(*res))

# plot results
fig, ax = plt.subplots()
ax.plot(h*T, St, 'b', label='Susceptible')
ax.plot(h * T, It, 'r', label='Infectious')
# ax.plot(h*T,Rt, 'g', label='Recovered')
ax.legend()
fig.savefig(f'Data/It_a')