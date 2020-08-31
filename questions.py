from numpy.random import shuffle

class CoherenceQuestion:
    def __init__(self, query, neighbor1, neighbor2, intruder):
        self.query = query
        self.neighbor1 = neighbor1
        self.neighbor2 = neighbor2
        self.intruder = intruder

    def index(self):
        if hasattr(self, "index2choice") and len(self.index2choice) == 4:
            return self.index2choice
        indecies = ["a", "b", "c", "d"]
        choices = [self.query, self.neighbor1, self.neighbor2, self.intruder]
        shuffle(indecies)
        self.index2choice = dict(zip(indecies, choices))
        return self.index2choice

    def is_answer_correct(self):
        if hasattr(self, "answer"):
            return self.intruder == self.index2choice[self.answer]

    def __str__(self):

        if hasattr(self, "index2choice") and len(self.index2choice) == 4:
            m = max((len(v) for v in self.index2choice.values()))
            to_show = "(a){:{m}s}\t\t(b){:{m}s}\n(c){:{m}s}\t\t(d){:{m}s}".format(self.index2choice['a'],
                                                                                  self.index2choice['b'],
                                                                                  self.index2choice['c'],
                                                                                  self.index2choice['d'], m=m)
            return to_show


class RelatednessQuestion:
    def __init__(self, query, candidate, type=None):
        self.query = query
        self.candidate = candidate
        self.type = type
        self.score = None

    def __str__(self):
        m = 30
        to_show = "{:{m}s}\t\t{:{m}s}".format(self.query, self.candidate, m=m)
        return to_show