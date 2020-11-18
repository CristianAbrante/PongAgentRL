class Agent(object):
    def __init__(self, player_id=1):
        # Set the player id that determines on which side the ai is going to play
        self.player_id = player_id
        # Ball prediction error, introduce noise such that SimpleAI reflects not
        # only in straight lines
        self.bpe = 4
        self.name = "dope agent"

    def load_model(self):
        return

    def get_name(self):
        """
        Interface function to retrieve the agents name
        """
        return self.name

    def get_action(self, ob=None):
        return 0

    def reset(self):
        # Nothing to done for now...
        return
