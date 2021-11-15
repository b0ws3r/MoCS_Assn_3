# Import networkx and a network file
import random
import networkx as nx
# import wget as wget
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def preferential_attachment(G, stubs):
  #add new node
  new_node = G.number_of_nodes()
  G.add_node(new_node)
  #pick attachment
  neighbor = np.random.choice(stubs)
  #add edge
  G.add_edge(new_node, neighbor)
  #update degrees
  G.nodes[new_node]['state'] = 1
  G.nodes[neighbor]['state'] = 1+G.nodes[neighbor]['state']
  stubs.append(new_node)
  stubs.append(neighbor)
  return(G, stubs)


def Get_Configuration_Model():
    G = nx.Graph()
    degrees = []

    # initial conditions
    G.add_node(0)
    G.add_node(1)
    G.add_edge(0, 1)
    degrees.append(0)
    degrees.append(1)
    G.nodes[0]['state'] = 1
    G.nodes[1]['state'] = 1

    for step in range(500):
        (G, degrees) = preferential_attachment(G, degrees)

    # delete the network but keep degrees
    G = nx.create_empty_copy(G)

    # rewire the network
    # shuffle degrees
    np.random.shuffle(degrees)
    firststubs = degrees[0::2]
    secondstubs = degrees[1::2]
    # connect degrees by pairs
    for e in range(len(firststubs)):
        G.add_edge(firststubs[e], secondstubs[e])
    return G


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
h = .001  # timestep
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
    res.append((S / configuration_model.number_of_nodes(), I / configuration_model.number_of_nodes()))

# zip unpacked list of tuples (n-th elements all together)
# map them to arrays
St, It = map(np.array, zip(*res))

# plot results
fig, ax = plt.subplots()
# ax.plot(h*T, St, 'b', label='Susceptible')
ax.plot(h * T, It, 'r', label='Infectious')
# ax.plot(h*T,Rt, 'g', label='Recovered')
ax.legend()
rhoStr = str(rho).replace('.', '')
fig.savefig(f'Data/It_b_{rhoStr}')