import tree

class DecisionTree:
    def __init__(self, test_data, gain_type, max_depth):
        self.data = test_data
        self.max_depth = max_depth

    def info_gain(self):
        return 1