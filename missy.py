# Import networkx and a network file
import random

import networkx as nx
# import wget as wget
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def Get_Configuration_Model():
    global G, degree_freq, degrees, adj_list
    # file = wget.download('http://snap.stanford.edu/data/as20000102.txt.gz')
    G = nx.read_edgelist('/home/missy/Documents/ComplexSystems/MoCS/Assn3/MoCS_Assn_3/Data/as20000102.txt')

    stublist = []
    ind = 0
    for deg in range(len(degree_freq) - 1):
        for i in range(degree_freq[deg]):
            seq = [ind] * deg
            stublist.extend(seq)
            ind = ind + 1
    adj_list = list()
    random.shuffle(stublist)
    for i in range(len(stublist) - 1):
        if i % 2 == 0:
            adj_list.append((stublist[i], stublist[i + 1]))

    df = pd.DataFrame.from_records(adj_list, columns=['start', 'edge'])
    df.to_csv('Data/stubs.csv')
    configuration_model = nx.MultiGraph()  # nx.parse_edgelist(adj_list)
    configuration_model.add_edges_from(adj_list)
    return configuration_model


configuration_model = Get_Configuration_Model()
degree_freq = nx.degree_histogram(configuration_model)
degrees = range(len(degree_freq))

plt.figure(figsize=(8, 6))
plt.loglog(degrees[1:], degree_freq[1:], 'ko-')
plt.xlabel('Degree')
plt.ylabel('Counts')

# Parameters of the model
beta = 0.3 # transmission rate
alpha = 1.0  # recovery rate

rho = 0.4
I0 = 0.01  # initial fraction of infected nodes
S0 = 1.0 - I0
Iv0 = 0
Sv0 = 0


# Initial conditions
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
h = 0.1  # timestep
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
    # R = 0.0

    # Run Euler's method for all degree classes k
    for k in range(len(degree_freq)):
        # calculate speeds
        delta_Sk = -beta * k * theta * Sk[k]
        delta_Ik = beta * k * theta * Sk[k] - alpha * Ik[k]
        # delta_Rk = alpha*Ik[k]
        # update dynamical variables
        Sk[k] += delta_Sk * h  # Ik(t+h)
        Ik[k] += delta_Ik * h  # Ik(t+h)
        # Rk[k] += delta_Rk*h #R(t+1)
        # update global quantities
        S += Sk[k]
        I += Ik[k]
        # R += Rk[k]
    res.append((S / G.number_of_nodes(), I / G.number_of_nodes()))

# zip unpacked list of tuples (n-th elements all together)
# map them to arrays
St, It = map(np.array, zip(*res))

# plot results
fig, ax = plt.subplots()
# ax.plot(h*T, St, 'b', label='Susceptible')
ax.plot(h * T, It, 'r', label='Infectious')
# ax.plot(h*T,Rt, 'g', label='Recovered')
ax.legend()
