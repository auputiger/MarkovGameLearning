from soccer.solver import Solver

# Hyper parameters
gamma = 0.9
alpha = 0.2
time_steps = 1000000

# Normal Q-learning test
solver = Solver()
stats = solver.q_learning(time_steps, alpha, gamma)

file = open('q-learning.csv', 'w')
for ts, q_diff, pre_q, post_q in stats:
    file.write('{},{},{},{}\n'.format(ts, q_diff, pre_q, post_q))

# Friend-Q test
solver = Solver()
stats = solver.friend_q_learning(time_steps, alpha, gamma)

file = open('friend-q.csv', 'w')
for ts, q_diff, pre_q, post_q in stats:
    file.write('{},{},{},{}\n'.format(ts, q_diff, pre_q, post_q))

# Foe-Q test
solver = Solver()
stats = solver.foe_q_learning(time_steps, alpha, gamma)

file = open('foe-q.csv', 'w')
for ts, q_diff, pre_q, post_q, probs in stats:
    file.write('{},{},{},{},{}\n'.format(ts, q_diff, pre_q, post_q, probs))

# CE-Q test
solver = Solver()
stats = solver.ce_q_learning(time_steps, alpha, gamma)

file = open('ce-q.csv', 'w')
for ts, q_diff, pre_q, post_q, probs in stats:
    file.write('{},{},{},{},{}\n'.format(ts, q_diff, pre_q, post_q, probs))
