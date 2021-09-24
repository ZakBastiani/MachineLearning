import numpy


class Node:
    def __init__(self, _attribute):
        self.attribute = _attribute
        self.branches = []


class Tree:
    def __init__(self, root_attribute):
        self.root = Node(root_attribute)
        self.current = self.root