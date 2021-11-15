# Import networkx and a network file
import math
import random

import networkx as nx
# import wget as wget
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def Get_Configuration_Model(degree_freq):
    # file = wget.download('http://snap.stanford.edu/data/as20000102.txt.gz')
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
    # df = pd.DataFrame.from_records(adj_list, columns=['start', 'edge'])
    # df.to_csv('Data/stubs.csv')
    configuration_model = nx.MultiGraph()  # nx.parse_edgelist(adj_list)
    configuration_model.add_edges_from(adj_list)
    return configuration_model


def uniform_attachment(G):
    # pick attachment
    nodes = 1 if G.number_of_nodes() == 0 else G.number_of_nodes()
    weights = np.random.geometric(0.25, nodes)
    neighbor = random.choices(G.nodes, weights=np.random.geometric(0.25, weights), k=1)
    # add new node
    new_node = G.number_of_nodes()
    G.add_node(new_node)
    # pick attachment
    # add edge
    G.add_edge(new_node, neighbor)
    # update degrees
    G.nodes[new_node]['state'] = 1
    G.nodes[neighbor]['state'] = 1 + G.nodes[neighbor]['state']
    return (G)


def graph_exponential_dist():
    z = [int(np.random.exponential(scale=5)) for i in range(1000)]
    if sum(z)%2 != 0:
        z[0] += 1
    configuration_model = nx.configuration_model(z)  # Get_Configuration_Model(z)
    return configuration_model


def graph_geometric_dist():
    z = [int(np.random.geometric(p=0.25)) for i in range(1000)]
    if sum(z)%2 != 0:
        z[0] += 1
    configuration_model = nx.configuration_model(z) # Get_Configuration_Model(z)
    return configuration_model


def og_initial_conditions(degree_freq):
    I0 = 0.01  # initial fraction of infected nodes
    Iv0 = 0.00
    Sv0 = 0.4
    S0 = 1.0 - I0 - Sv0 - Iv0
    Sk = np.zeros((len(degree_freq)))  # array for expected S_k
    Ik = np.zeros((len(degree_freq)))  # array for expected I_k
    Svk = np.zeros((len(degree_freq)))  # array for expected I_k
    Ivk = np.zeros((len(degree_freq)))  # array for expected I_k

    for k in range(len(degree_freq)):
        # set an expectation of I0 fraction of nodes of degree k infectious
        Sk[k] = degree_freq[k] * S0
        Svk[k] = degree_freq[k] * Sv0
        Ik[k] = degree_freq[k] * I0
        Ivk[k] = degree_freq[k] * Iv0
    return Sk, Svk, Ik, Ivk, S0, Sv0, I0, Iv0


def part_d_initial_conditions(degree_freq):
    I0 = 0.01  # initial fraction of infected nodes
    Iv0 = 0.00
    k_60_0 = np.ceil(0.6 * len(degree_freq))
    cnt = 0

    degrees_by_node = [val for (node, val) in configuration_model.degree()]
    # find degrees by node where node degree is in top 40%
    for i in range(len(degrees_by_node)):
        if degrees_by_node[i] > k_60_0:
            cnt += 1
    Sv0 = (cnt / configuration_model.number_of_nodes())
    S0 = 1.0 - I0 - Sv0 - Iv0

    Sk = np.zeros((len(degree_freq)))  # array for expected S_k
    Ik = np.zeros((len(degree_freq)))  # array for expected I_k
    Svk = np.zeros((len(degree_freq)))  # array for expected I_k
    Ivk = np.zeros((len(degree_freq)))  # array for expected I_k

    for k in range(len(degree_freq)):
        # set an expectation of I0 fraction of nodes of degree k infectious
        Sk[k] = degree_freq[k] * S0
        if k > k_60_0:
            Svk[k] = 1 #degree_freq[k]
        Ik[k] = degree_freq[k] * I0
        Ivk[k] = degree_freq[k] * Iv0
    return Sk, Svk, Ik, Ivk, S0, Sv0, I0, Iv0


# configuration_model = graph_exponential_dist()
configuration_model = graph_geometric_dist()

# gather some info about the model
degree_freq = nx.degree_histogram(configuration_model)
degrees = range(len(degree_freq))
G_deg_sum = [a * b for a, b in zip(degree_freq, range(0, len(degree_freq)))]
total_degree = sum(G_deg_sum)

avg_k = sum(G_deg_sum) / configuration_model.number_of_nodes()
print(avg_k)

# set up deg freq plot if we want
plt.figure(figsize=(8, 6))
plt.loglog(degrees[1:], degree_freq[1:], 'ko-')
plt.xlabel('Degree')
plt.ylabel('Counts')

# Parameters of the model
alpha = .10
beta = .03  # transmission rate
rho = 0.1
rho = 0.001

# Initial conditions
# Sk, Svk, Ik, Ivk, S0, Sv0, I0, Iv0 = og_initial_conditions(degree_freq)
Sk, Svk, Ik, Ivk, S0, Sv0, I0, Iv0  = part_d_initial_conditions(degree_freq)

# Run the model
# Discrete steps of Euler's methods
res = []  # list of results
history = []
S = S0
I = I0 # set initial conditions
Iv = Iv0
Sv = Sv0
h = .01  # timestep
T = np.arange(1, 500 / h)
for t in T:

    # Calculate the mean-field
    theta = 0.0
    for k in range(len(degree_freq)):
        theta += k * (Ik[k] + Ivk[k]) / total_degree
    history.append(theta)

    # Set initial global quantities
    S = 0.0
    I = 0.0
    Iv = 0.0
    Sv = 0.0

    # Run Euler's method for all degree classes k
    for k in range(len(degree_freq)):
        # calculate speeds+ alpha * Ik[k]
        delta_Sk = -beta * k * theta * Sk[k] + alpha * Ik[k]
        delta_Svk = (1 - rho) * -beta * k * theta * Ivk[k] + alpha * Ivk[k]
        delta_Ik = beta * k * theta * Sk[k] - alpha * Ik[k]
        delta_Ivk = (1 - rho) * beta * k * theta * Svk[k] - alpha * Ivk[k]

        # update dynamical variables
        Sk[k] += delta_Sk * h  # Ik(t+h)
        Svk[k] += delta_Svk * h  # Ik(t+h)
        Ik[k] += delta_Ik * h  # Ik(t+h)
        Ivk[k] += delta_Ivk * h

        # update global quantities
        S += Sk[k]
        Sv += Svk[k]
        I += Ik[k]
        Iv += Ivk[k]
    res.append((S / configuration_model.number_of_nodes(),
                I / configuration_model.number_of_nodes(),
                Sv / configuration_model.number_of_nodes(),
                Iv / configuration_model.number_of_nodes()
                )
               )

# zip unpacked list of tuples (n-th elements all together)
# map them to arrays
St, It, Svt, Ivt = map(np.array, zip(*res))

# plot results
fig, ax = plt.subplots()
ax.plot(h * T, St, label='Susceptible')
ax.plot(h * T, It, label='Infectious')
ax.plot(h * T, Ivt, label='Infectious Vaxxed')
ax.plot(h * T, Svt, label='Susceptible Vaxxed')
# ax.plot(h*T,Rt, 'g', label='Recovered')
ax.legend()
fig.show()
fig.savefig('Data/It')
