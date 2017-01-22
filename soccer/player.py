class Player:
    """
    Soccer player
    """

    def __init__(self, num, cords, has_ball):
        """
        :param num: player number, should be unique
        :param cords: coordinates of player in the soccer field. Two players
        should not occupy the same space.
        :param has_ball: boolean, True if player has possession of the ball.
        """
        self.num = num
        self.cords = cords
        self.has_ball = has_ball

    def __eq__(self, other):
        return self.num == other.num and self.cords == other.cords \
               and self.has_ball == other.has_ball

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash((self.num, self.cords, self.has_ball))

    def __str__(self):
        return str((self.num, self.cords, self.has_ball))
