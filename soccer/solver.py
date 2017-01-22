import copy
import random
from collections import defaultdict

from cvxopt.modeling import op
from cvxopt.modeling import variable
from cvxopt.solvers import options

from soccer.actions import Actions
from soccer.player import Player
from soccer.soccer_game import SoccerGame
from soccer.state import State


class Solver:
    def __init__(self):
        # Actions players can take
        self.actions = [Actions.UP, Actions.DOWN, Actions.RIGHT, Actions.LEFT,
                        Actions.STICK]

        # Starting game state for the soccer match
        self.init_state = State(Player(1, (3, 0), False),
                                Player(2, (1, 0), True))

        # State for gather Q-value differences
        self.q_stat_state = State(Player(1, (2, 0), False),
                                  Player(2, (1, 0), True))

        # V tables for players A and B initialized to 1
        self.V1 = defaultdict(lambda: 1)
        self.V2 = defaultdict(lambda: 1)

    def q_learning(self, time_steps, alpha, gamma):
        # List of gathered statistics
        statistics = list()

        # State action pair to gather q-value differences from
        q_stat = (self.q_stat_state, Actions.DOWN)

        # Q-table
        Q = defaultdict(lambda: 1)

        # Time step counter
        time_step_counter = 0

        # Soccer game
        game = SoccerGame(self.init_state)
        game_over = False

        while time_step_counter < time_steps:
            # Restart game if ended already.
            if game_over:
                game = SoccerGame(self.init_state)
                game_over = False

            # Current state
            cur_state = copy.deepcopy(game.state)

            # Select actions
            a = random.choice(self.actions)
            o = random.choice(self.actions)

            # Apply actions
            game.apply_actions(a, o)
            time_step_counter += 1

            # log time step and alpha
            if time_step_counter % 10000 == 0:
                print(time_step_counter)
                print(alpha)

            # Get reward
            current_reward = game.state.reward_value()

            # If reward not zero, game is now over.
            if current_reward != 0:
                game_over = True

            # Set value of new state
            self.V1[game.state] = max(Q[(game.state, Actions.UP)],
                                      Q[(game.state, Actions.DOWN)],
                                      Q[(game.state, Actions.RIGHT)],
                                      Q[(game.state, Actions.LEFT)],
                                      Q[(game.state, Actions.STICK)])

            # previous q-value
            pre_q = Q[(cur_state, a)]

            # q-update
            Q[(cur_state, a)] = (1 - alpha) * Q[(cur_state, a)] + alpha * (
                current_reward + gamma * self.V1[game.state])

            # post update q-value
            post_q = Q[(cur_state, a)]

            # record stats if in correct state action pair
            if (cur_state, a) == q_stat:
                statistics.append(
                    (time_step_counter, abs(post_q - pre_q), pre_q, post_q))

            # decay alpha
            alpha = max(0.001, alpha * 0.999995)

        # format is (time-step, q-diff, pre-q-val, post-q-val)
        return statistics

    def friend_q_learning(self, time_steps, alpha, gamma):
        # gathered stats
        statistics = list()

        # state joint action pair to record q-diff's
        q_stat = (self.q_stat_state, Actions.DOWN, Actions.STICK)

        # Q-table
        Q = defaultdict(lambda: 1)

        # time-step counter
        time_step_counter = 0

        # Start soccer game
        game = SoccerGame(self.init_state)
        game_over = False

        while time_step_counter < time_steps:
            # restart game if needed
            if game_over:
                game = SoccerGame(self.init_state)
                game_over = False

            # current state
            cur_state = copy.deepcopy(game.state)

            # select actions
            a = random.choice(self.actions)
            o = random.choice(self.actions)

            # take actions
            game.apply_actions(a, o)
            time_step_counter += 1

            # log time-step and alpha
            if time_step_counter % 10000 == 0:
                print(time_step_counter)
                print(alpha)

            # get current reward
            current_reward = game.state.reward_value()

            # if not 0, game is over
            if current_reward != 0:
                game_over = True

            # get max q-value
            max_q_value = Q[(game.state, Actions.UP, Actions.UP)]
            for p1_a in self.actions:
                for p2_o in self.actions:
                    max_q_value = max(max_q_value, Q[(game.state, p1_a, p2_o)])

            # update value of state
            self.V1[game.state] = max_q_value

            # pre q-value
            pre_q = Q[(cur_state, a, o)]

            # q-value update
            Q[(cur_state, a, o)] = (1 - alpha) * Q[
                (cur_state, a, o)] + alpha * (
                current_reward + gamma * self.V1[game.state])

            # post q-value
            post_q = Q[(cur_state, a, o)]

            # record q-diff's
            if (cur_state, a, o) == q_stat:
                statistics.append(
                    (time_step_counter, abs(post_q - pre_q), pre_q, post_q))

            # decay alpha
            alpha = max(0.001, alpha * 0.999995)

        # format (time-step, q-diff, pre-q-val, post-q-val)
        return statistics

    def foe_q_learning(self, time_steps, alpha, gamma):
        # turn lp solver logging off
        options['show_progress'] = False

        # gathered stats
        statistics = list()

        # state joint action pair to record q-diffs in
        q_stat = (self.q_stat_state, Actions.DOWN, Actions.STICK)

        # Q-table
        Q = defaultdict(lambda: 1)

        # time-step counter
        time_step_counter = 0

        # Start game
        game = SoccerGame(self.init_state)
        game_over = False

        while time_step_counter < time_steps:
            # restart game if needed
            if game_over:
                game = SoccerGame(self.init_state)
                game_over = False

            # current state
            cur_state = copy.deepcopy(game.state)

            # select actions
            a = random.choice(self.actions)
            o = random.choice(self.actions)

            # apply actions
            game.apply_actions(a, o)
            time_step_counter += 1

            # log info
            if time_step_counter % 10000 == 0:
                print(time_step_counter)
                print(alpha)

            # get current reward
            current_reward = game.state.reward_value()
            if current_reward != 0:
                game_over = True

            # action probabilities
            probs = list()
            for i in range(len(self.actions)):
                probs.append(variable())

            # all action probabilities >= 0 constraints
            constrs = list()
            for i in range(len(self.actions)):
                constrs.append((probs[i] >= 0))

            # sum of probabilities = 1 constraint
            total_prob = sum(probs)
            constrs.append((total_prob == 1))

            # objective
            v = variable()

            # set mini-max constraints
            for j in range(5):
                c = 0
                for i in range(5):
                    c += Q[(game.state, self.actions[i], self.actions[j])] * \
                         probs[i]
                constrs.append((c >= v))

            # maximize objective
            lp = op(-v, constrs)
            lp.solve()

            # set value of state
            max_q_value = v.value[0]
            self.V1[game.state] = max_q_value

            # pre q-value
            pre_q = Q[(cur_state, a, o)]

            # update q-value
            Q[(cur_state, a, o)] = (1 - alpha) * Q[
                (cur_state, a, o)] + alpha * (
                current_reward + gamma * self.V1[game.state])

            # post q-value
            post_q = Q[(cur_state, a, o)]

            # gather statistics
            if (cur_state, a, o) == q_stat:
                prob_list = list()
                for i in range(len(probs)):
                    prob_list.append(probs[i].value[0])
                statistics.append((
                                  time_step_counter, abs(post_q - pre_q), pre_q,
                                  post_q, prob_list))

            # decay alpha
            alpha = max(0.001, alpha * 0.999995)

        # format (time-step, q-diff, pre-q-val, post-q-val, probabilities)
        return statistics

    def ce_q_learning(self, time_steps, alpha, gamma):
        # Turn off lp logging
        options['show_progress'] = False

        # gathered stats
        statistics = list()
        q_stat = (self.q_stat_state, Actions.DOWN, Actions.STICK)

        # Q-tables for each player
        Q1 = defaultdict(lambda: 1)
        Q2 = defaultdict(lambda: 1)

        # time-step counter
        time_step_counter = 0

        # start game
        game = SoccerGame(self.init_state)
        game_over = False

        while time_step_counter < time_steps:
            # restart game if needed
            if game_over:
                game = SoccerGame(self.init_state)
                game_over = False

            # current state
            cur_state = copy.deepcopy(game.state)

            # choose actions
            a = random.choice(self.actions)
            o = random.choice(self.actions)

            # apply actions
            game.apply_actions(a, o)
            time_step_counter += 1

            # log info
            if time_step_counter % 1000 == 0:
                print(time_step_counter)
                print(alpha)

            # get current reward
            current_reward = game.state.reward_value()
            if current_reward != 0:
                game_over = True

            # joint action probabilities
            probs = {}
            for i in range(len(self.actions)):
                for j in range(len(self.actions)):
                    probs[(i, j)] = variable()

            # all joint action probabilities >= 0 constraints
            constrs = list()
            for i in range(len(self.actions)):
                for j in range(len(self.actions)):
                    constrs.append((probs[(i, j)] >= 0))

            # sum of joint action probabilities = 1
            total_prob = 0
            for i in range(len(self.actions)):
                for j in range(len(self.actions)):
                    total_prob += probs[(i, j)]
            constrs.append((total_prob == 1))

            # objective
            v = variable()

            # 20 rationality constraints for player A
            for i in range(len(self.actions)):
                rc1 = 0
                for j in range(len(self.actions)):
                    rc1 += probs[(i, j)] * Q1[
                        (game.state, self.actions[i], self.actions[j])]

                for k in range(len(self.actions)):
                    if i != k:
                        rc2 = 0
                        for l in range(len(self.actions)):
                            rc2 += probs[(i, l)] * Q1[
                                (game.state, self.actions[k], self.actions[l])]
                        constrs.append((rc1 >= rc2))

            # 20 rationality constraints for player B
            for i in range(len(self.actions)):
                rc1 = 0
                for j in range(len(self.actions)):
                    rc1 += probs[(j, i)] * Q2[
                        (game.state, self.actions[j], self.actions[i])]

                for k in range(len(self.actions)):
                    if i != k:
                        rc2 = 0
                        for l in range(len(self.actions)):
                            rc2 += probs[(l, i)] * Q2[
                                (game.state, self.actions[l], self.actions[k])]
                        constrs.append((rc1 >= rc2))

            # sum of the players rewards
            sum_total = 0
            for i in range(len(self.actions)):
                for j in range(len(self.actions)):
                    sum_total += probs[(i, j)] * Q1[
                        (game.state, self.actions[i], self.actions[j])]
                    sum_total += probs[(i, j)] * Q2[
                        (game.state, self.actions[i], self.actions[j])]

            constrs.append((v == sum_total))

            # maximize sum of players rewards
            lp = op(-v, constrs)
            lp.solve()

            if lp.status == 'optimal':
                # update V table for player A
                v1 = 0
                for i in range(len(self.actions)):
                    for j in range(len(self.actions)):
                        v1 += probs[(i, j)].value[0] * Q1[
                            (game.state, self.actions[i], self.actions[j])]
                self.V1[game.state] = v1

                # update V table for player B
                v2 = 0
                for i in range(len(self.actions)):
                    for j in range(len(self.actions)):
                        v2 += probs[(i, j)].value[0] * Q2[
                            (game.state, self.actions[i], self.actions[j])]
                self.V2[game.state] = v2

            # pre q-value
            pre_q = Q1[(cur_state, a, o)]

            # P1 Q update
            Q1[(cur_state, a, o)] = (1 - alpha) * Q1[
                (cur_state, a, o)] + alpha * (
                current_reward + gamma * self.V1[game.state])

            # P2 Q Update
            Q2[(cur_state, a, o)] = (1 - alpha) * Q2[
                (cur_state, a, o)] + alpha * (
                -current_reward + gamma * self.V2[game.state])

            # post q-value
            post_q = Q1[(cur_state, a, o)]

            # gather stats
            if (cur_state, a, o) == q_stat:
                prob_list = list()
                for x in range(5):
                    for y in range(5):
                        prob_list.append(probs[(x, y)].value[0])
                statistics.append((
                                  time_step_counter, abs(post_q - pre_q), pre_q,
                                  post_q, prob_list))

            # decay alpha
            alpha = max(0.001, alpha * 0.999995)

        # format (time-step, q-diff, pre-q-val, post-q-val, probabilities)
        return statistics
