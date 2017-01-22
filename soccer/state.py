from soccer.actions import Actions
from soccer.player import Player


class State:
    # Dimensions of grid soccer field.
    FIELD_DIMENSIONS = [4, 2]

    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2

    def get_reachable_states(self, a, o):
        """
        :param a: player 1's action
        :param o: player 2's action
        :return: Set of possible states after actions are executed.
        """
        reachable_states = set()

        # If player A moves first
        new_cords_a = self.new_player_cords(self.player1, a)
        new_cords_b = self.new_player_cords(self.player2, o)

        if new_cords_a == self.player2.cords:
            if self.player1.has_ball:
                tmp_a = Player(1, self.player1.cords, False)
                tmp_b = Player(2, self.player2.cords, True)
            else:
                tmp_a = Player(1, self.player1.cords, False)
                tmp_b = Player(2, self.player2.cords, True)
        else:
            if self.player1.has_ball:
                tmp_a = Player(1, new_cords_a, True)
                if new_cords_a == new_cords_b:
                    tmp_b = Player(2, self.player2.cords, False)
                else:
                    tmp_b = Player(2, new_cords_b, False)
            else:
                if new_cords_a == new_cords_b:
                    tmp_a = Player(1, new_cords_a, True)
                    tmp_b = Player(2, self.player2.cords, False)
                else:
                    tmp_a = Player(1, new_cords_a, False)
                    tmp_b = Player(2, new_cords_b, True)

        reachable_states.add(State(tmp_a, tmp_b))

        # If player B moves first
        new_cords_a = self.new_player_cords(self.player1, a)
        new_cords_b = self.new_player_cords(self.player2, o)
        if new_cords_b == self.player1.cords:
            if self.player2.has_ball:
                tmp_b = Player(2, self.player2.cords, False)
                tmp_a = Player(1, self.player1.cords, True)
            else:
                tmp_b = Player(2, self.player2.cords, False)
                tmp_a = Player(1, self.player1.cords, True)
        else:
            if self.player2.has_ball:
                tmp_b = Player(2, new_cords_b, True)
                if new_cords_b == new_cords_a:
                    tmp_a = Player(1, self.player1.cords, False)
                else:
                    tmp_a = Player(1, new_cords_a, False)
            else:
                if new_cords_b == new_cords_a:
                    tmp_b = Player(2, new_cords_b, True)
                    tmp_a = Player(1, self.player1.cords, False)
                else:
                    tmp_b = Player(2, new_cords_b, False)
                    tmp_a = Player(1, new_cords_a, True)

        reachable_states.add(State(tmp_a, tmp_b))

        return reachable_states

    def __eq__(self, other):
        return self.player1 == other.player1 and self.player2 == other.player2

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.player1, self.player2))

    def __str__(self):
        return str((str(self.player1), str(self.player2)))

    def reward_value(self):
        """
        :return: The reward for player 1, since its a zero sum game, player 2's
        reward is the negation of the reward.
        """
        if self.player1.has_ball:
            x, y = self.player1.cords
            if x == 0:
                return 100
            elif x == State.FIELD_DIMENSIONS[0] - 1:
                return -100
            else:
                return 0
        elif self.player2.has_ball:
            x, y = self.player2.cords
            if x == 0:
                return 100
            elif x == State.FIELD_DIMENSIONS[0] - 1:
                return -100
            else:
                return 0

    @staticmethod
    def new_player_cords(player, action):
        """
        :param player: Some player
        :param action: Action player has selected to take
        :return: tuple of coordinates where new player will be if action is
        successful.
        """
        x, y = player.cords
        if action == Actions.UP:
            y = max(0, y - 1)
        elif action == Actions.DOWN:
            y = min(State.FIELD_DIMENSIONS[1] - 1, y + 1)
        elif action == Actions.LEFT:
            x = max(0, x - 1)
        elif action == Actions.RIGHT:
            x = min(State.FIELD_DIMENSIONS[0] - 1, x + 1)
        return x, y
