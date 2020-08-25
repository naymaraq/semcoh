import argparse
from nearest_neighbors import nn
from numpy.random import shuffle

class Question:

    def __init__(self, query, neighbor1, neighbor2, intruder):
        self.query = query
        self.neighbor1 = neighbor1
        self.neighbor2 = neighbor2
        self.intruder = intruder

    def index(self):
        if hasattr(self, "index2choice") and len(self.index2choice)==4:
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

        if hasattr(self, "index2choice") and len(self.index2choice)==4:
            m = max((len(v) for v in self.index2choice.values()))
            to_show = "(a){:{m}s}\t\t(b){:{m}s}\n(c){:{m}s}\t\t(d){:{m}s}".format(self.index2choice['a'],
                                                                              self.index2choice['b'],
                                                                              self.index2choice['c'],
                                                                              self.index2choice['d'], m = m)
            return to_show
            

def get_answer():
    answer = input("Type the answer [a-d,n]:")
    if answer in ["a", "b", "c", "d", "n"]:
        return answer
    return get_answer()

class Survey:

    def __init__(self, vectors_path, vocab_path, freqs, normalize):
        self.nn_obj = nn(vectors_path, vocab_path, normalize=normalize)
        self.questions = []

    def survey_step(self):
        query = self.nn_obj.rand_query()
        neighbors, intruder= self.nn_obj.rand_neighbors(query, 2, 0.7)
        
        q = None
        if any(neighbors) and intruder:
            q = Question(query, neighbors[0], neighbors[1], intruder)
            q.index()
        return q

    def start_survey(self):
        try:
            while True:
                q = self.survey_step()
                if q:
                    print('-'*100)
                    print(q)
                    answer = get_answer()
                    if answer == 'n':
                        continue
                    else:
                        q.answer = answer
                        self.questions.append(q)

        except KeyboardInterrupt:
            if len(self.questions):
                result = [q.is_answer_correct() for q in self.questions]
                print()
                print(f"Intruision detection rate: {sum(result)*100/len(result)}")
                print()


ARG = argparse.ArgumentParser()
ARG.add_argument('--v', type=str, help='Path to vectors.npy')
ARG.add_argument('--i', type=str, help='Path to index.json')
ARG.add_argument('--f', type=str, default='', help='Path to freqs.json')
ARG.add_argument('--n', type=bool, default=True, help='Normalize')

ARG = ARG.parse_args()


survey = Survey(ARG.v, ARG.i, ARG.f, ARG.n)
survey.start_survey()




