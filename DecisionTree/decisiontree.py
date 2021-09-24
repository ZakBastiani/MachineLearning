import math
import pandas as pd


class Node:
    def __init__(self, _attribute):
        self.attribute = _attribute
        self.branches = []


class DecisionTree:
    def __init__(self, train_data, gain_type, max_depth):
        self.data = train_data
        self.max_depth = max_depth
        self.root = self.buildtree(self.data)

    def buildtree(self, current_Set):
        # if current set has the same label set this as a leaf with the aforementioned label
        current_Set

        # Find the best attribute

        # Set that as the attribute for the root
        root = Node(best_attribute)
        # Split the branches into subsets for each attribute label
        # If a label is empty set it as the most common label inside of this attribute
        # Build a tree for every branch

        return root

    def IG(self, set_A, attribute):
        return 1

    def ME(self, set_A, attribute):
        return 1

    def GI(self, set_A, attribute):
        return 1