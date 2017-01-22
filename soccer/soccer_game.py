import copy
import random


class SoccerGame:
    """
    Soccer game. Intended only for a 2 player match.
    """

    def __init__(self, state):
        """
        :param state: State the game should start in.
        """
        self.state = copy.deepcopy(state)

    def apply_actions(self, a, o):
        """
        Both players choose actions simultaneously, however actions are not
        executed simultaneously. There is a 50% chance player 1 will go before
        player 2.
        :param a: Action player 1 is attempting.
        :param o: Action player 2 is attempting.
        :return: None
        """
        self.state = random.sample(self.state.get_reachable_states(a, o), 1)[0]
