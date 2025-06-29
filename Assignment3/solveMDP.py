import numpy as np


# read mdp files and extract parameters
def parse_mdp_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    transitions = {}
    end_states = []
    num_states = num_actions = 0
    mdptype = ""
    discount = 0.0

    for line in lines:
        parts  = line.strip().split()

        if parts[0] == "numStates":
            num_states = int(parts[1])

        elif parts[0] == "numActions":
            num_actions = int(parts[1])

        elif parts[0] == "end":
            if parts[1] != "-1":
                end_states = list(map(int, parts[1:]))

        elif parts[0] == "transition":
            s, a, s2 =map(int, parts[1:4])
            r,p = map(float, parts[4:])
            if (s, a) not in transitions:
                transitions[(s, a)] = []
            transitions[(s,a)].append((s2, r,p))

        elif parts[0] == "mdptype":
            mdptype = parts [1]

        elif parts[0] == "discount":
            discount = float(parts[1])


    return num_states, num_actions, transitions, end_states, mdptype, discount


# value iteration algorithm
def value_iteration(num_states, num_actions, transitions, discount, end_states, threshold = 1e-10):
    V = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)

    while True:

        delta = 0
        new_V= np.copy(V)

        for s in range(num_states):
            if s in end_states:
                new_V[s] = 0
                policy[s] = -1  
                continue
            
            action_values = []
            for a in range(num_actions):
               total = 0
               
               for s2, r, p in transitions.get((s, a), []):
                total += p* (r + discount* V[s2])
                action_values.append(total)
            new_V[s] = max(action_values)
            policy[s] = np.argmax(action_values)
            delta = max(delta, abs(new_V[s]- V[s]))

        V = new_V
        if delta < threshold:
            break

    return V, policy


# driver code

# continuing-mdp-2-2.txt
filename = "data/continuing-mdp-2-2.txt"
S, A, transitions, end_states, mdptype, gamma = parse_mdp_file(filename)
V, policy = value_iteration(S, A, transitions, gamma, end_states)

print(f"Results for {filename}:")
for s in range(S):
    print(f"{V[s]:.6f} {policy[s]}")

# S continuing-mdp-10-5.txt
filename = "data/continuing-mdp-10-5.txt"
S, A, transitions, end_states, mdptype, gamma = parse_mdp_file(filename)
V, policy = value_iteration(S, A, transitions, gamma, end_states)

print(f"Results for {filename}:")
for s in range(S):
    print(f"{V[s]:.6f} {policy[s]}")


# \continuing-mdp-50-20.txt
filename = "data/continuing-mdp-50-20.txt"
S, A, transitions, end_states, mdptype, gamma = parse_mdp_file(filename)
V, policy = value_iteration(S, A, transitions, gamma, end_states)

print(f"Results for {filename}:")
for s in range(S):
    print(f"{V[s]:.6f} {policy[s]}")


# episodic-mdp-2-2.txt
filename = "data/episodic-mdp-2-2.txt"
S, A, transitions, end_states, mdptype, gamma = parse_mdp_file(filename)
V, policy = value_iteration(S, A, transitions, gamma, end_states)

print(f"Results for {filename}:")
for s in range(S):
    print(f"{V[s]:.6f} {policy[s]}")


# episodic-mdp-10-5.txt
filename = "data/episodic-mdp-10-5.txt"
S, A, transitions, end_states, mdptype, gamma = parse_mdp_file(filename)
V, policy = value_iteration(S, A, transitions, gamma, end_states)

print(f"Results for {filename}:")
for s in range(S):
    print(f"{V[s]:.6f} {policy[s]}")


# episodic-mdp-50-20.txt
filename = "data/episodic-mdp-50-20.txt"
S, A, transitions, end_states, mdptype, gamma = parse_mdp_file(filename)
V, policy = value_iteration(S, A, transitions, gamma, end_states)

print(f"Results for {filename}:")
for s in range(S):
    print(f"{V[s]:.6f} {policy[s]}")



        


    
        
                
                
            
        
      
        
    
    

