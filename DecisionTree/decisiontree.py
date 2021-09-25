import math
import pandas as pd


class Node:
    def __init__(self, _attribute):
        self.attribute = _attribute
        self.branches = []

class Branch:
    def __init__(self, att_type, node):
        self.att_type = att_type
        self.node = node


class DecisionTree:
    IG_ID = 0
    ME_ID = 1
    GI_ID = 2

    def __init__(self, train_data, attribute_types, gain_type, max_depth):
        self.data = train_data
        self.max_depth = max_depth
        self.gain = self.IG
        self.label = self.data.columns[-1]
        self.types = attribute_types
        self.root = self.buildtree(self.data, 1)

    def buildtree(self, current_Set, depth):
        attributes = current_Set.columns[:-1]
        # if current set has the same label set this as a leaf with the aforementioned label
        if current_Set[self.label].unique().size == 1:
            return Node(current_Set.iloc[0][self.label])

        if attributes.size == 0 or depth == self.max_depth:
            return Node(current_Set[self.label].mode()[0])

        # Find the best attribute
        best_value = 0
        best_attribute = ""
        for a in attributes:
            value = self.gain(current_Set, a)
            if value > best_value:
                best_value = value
                best_attribute = a

        # Set that as the attribute for the root
        root = Node(best_attribute)
        # Split the branches into subsets for each attribute label
        for tp in self.types[best_attribute]:
            # If a label is empty set it as the most common label inside of this attribute
            if current_Set[current_Set[best_attribute] == tp][self.label].count() == 0:
                root.branches.append(Branch(tp, Node(current_Set[self.label].mode()[0])))
            else:
                # Build a tree for every branch
                root.branches.append(Branch(tp, self.buildtree(current_Set[current_Set[best_attribute] == tp].drop(columns=best_attribute), depth+1)))

        return root

    def IG(self, set_A, attribute):
        output = 0
        for l in pd.unique(set_A[self.label]):
            prob = set_A[set_A[self.label] == l][self.label].count()/set_A[self.label].count()
            output += -prob*math.log2(prob)
        types = pd.unique(set_A[attribute])
        for t in types:
            set_ratio = set_A[set_A[attribute] == t][attribute].count()/set_A[attribute].count()
            sub_entropy = 0
            for l in pd.unique(set_A[self.label]):
                prob = set_A[set_A[attribute] == t][set_A[self.label] == l][self.label].count() /set_A[set_A[attribute] == t][attribute].count()
                if prob != 0:
                    sub_entropy += -prob*math.log2(prob)
            output -= set_ratio*sub_entropy
        return output

    def ME(self, set_A, attribute):
        output = 0
        me = []
        for l in pd.unique(set_A[self.label]):
            prob = set_A[set_A[self.label] == l][self.label].count()/set_A[self.label].count()
            me.append(-prob*math.log2(prob))
        output = me.sort()[1]
        types = pd.unique(set_A[attribute])
        for t in types:
            set_ratio = set_A[set_A[attribute] == t][attribute].count()/set_A[attribute].count()
            sub_me = []
            for l in pd.unique(set_A[self.label]):
                prob = set_A[set_A[attribute] == t][set_A[self.label] == l][self.label].count() /set_A[set_A[attribute] == t][attribute].count()
                sub_me.append(prob)
            output -= set_ratio*(sub_me.sort()[1])
        return output

    def GI(self, set_A, attribute):
        output = 1
        for l in pd.unique(set_A[self.label]):
            prob = set_A[set_A[self.label] == l][self.label].count()/set_A[self.label].count()
            output += -prob**2
        types = pd.unique(set_A[attribute])
        for t in types:
            set_ratio = set_A[set_A[attribute] == t][attribute].count()/set_A[attribute].count()
            sub_entropy = 1
            for l in pd.unique(set_A[self.label]):
                prob = set_A[set_A[attribute] == t][set_A[self.label] == l][self.label].count() /set_A[set_A[attribute] == t][attribute].count()
                if prob != 0:
                    sub_entropy += -prob**2
            output -= set_ratio*sub_entropy
        return output

    def testdata(self, test_data):
        acc = 0
        for i in range(0, test_data[self.label].count()):
            r = test_data.iloc[i]
            current = self.root
            while len(current.branches) != 0:
                att_type = r[current.attribute]
                for b in current.branches:
                    if b.att_type == att_type:
                        current = b.node
                        break

            if r[self.label] == current.attribute:
                acc += 1
        acc = acc/test_data[self.label].count()
        return acc
