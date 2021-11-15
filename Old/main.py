# Import networkx and a network file
import networkx as nx
# import wget as wget
import matplotlib.pyplot as plt
import numpy as np

# file = wget.download('http://snap.stanford.edu/data/as20000102.txt.gz')

G = nx.read_edgelist('/MoCS_Assn_3/Data/as20000102.txt')

degree_freq = nx.degree_histogram(G)
degrees = range(len(degree_freq))
plt.figure(figsize=(8, 6))
plt.loglog(degrees[1:], degree_freq[1:], 'ko-')
plt.xlabel('Degree')
plt.ylabel('Counts')
plt.show()

# Parameters of the model
beta = 0.025  # transmission rate
alpha = 0.01  # recovery rate

I0 = 0.01  # initial fraction of infected nodes
S0 = 1.0 - I0
# R0 = 0.0

# Initial conditions
G_deg_sum = [a * b for a, b in zip(degree_freq, range(0, len(degree_freq)))]
total_degree = sum(G_deg_sum)
avg_k = sum(G_deg_sum) / G.number_of_nodes()
print(avg_k)
Sk = np.zeros((len(degree_freq)))  # array for expected S_k
Ik = np.zeros((len(degree_freq)))  # array for expected I_k
# Rk = np.zeros((len(degree_freq))) #array for expected R_k

for k in range(len(degree_freq)):
    # set an expectation of I0 fraction of nodes of degree k infectious
    Sk[k] = degree_freq[k] * S0
    Ik[k] = degree_freq[k] * I0
    # Rk[k] = degree_freq[k]*R0

# Run the model

# Discrete steps of Euler's methods
res = []  # list of results
history = []
S = S0;
I = I0;  # set initial conditions
h = 0.1;  # timestep
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
