from nearest_neighbors import nn, RelatedPairs
from questions import CoherenceQuestion, RelatednessQuestion
from utils import get_answer, get_float_answer


class RelatednessSurvey:
    def __init__(self, query_list_path, relateded_pairs_path):
        self.rp_obj = RelatedPairs(query_list_path, relateded_pairs_path)
        self.questions = []

    def survey_step(self):
        pair = self.rp_obj.next_pair()
        q = None
        if any(pair):
            q = RelatednessQuestion(*pair)
        return q

    def start_survey(self):

        save_as = 'relatedness_survey.csv'

        try:
            q_index = 0
            q = self.survey_step()
            while q:
                print()
                print(f"Question {q_index}.")
                print(f'-' * 80)
                print(q)

                score = get_float_answer()
                if score == 'n':
                    continue
                else:
                    q.score = score
                    self.questions.append(q)
                q_index += 1
                q = self.survey_step()

            self.save(save_as=save_as)
        except KeyboardInterrupt:
            self.save(save_as=save_as)

    def save(self, save_as):
        print(f'Save the result in {save_as}')
        with open(save_as, 'w') as f:
            f.write("Dom1,Dom2,Type,Score\n")
            for q in self.questions:
                f.write(f"{q.query},{q.candidate},{q.type},{q.score}\n")


class CoherenceSurvey:
    def __init__(self, vectors_path, vocab_path, freqs, query_list_path, normalize):
        self.nn_obj = nn(vectors_path, vocab_path, query_list_path, normalize=normalize)
        self.questions = []

    def survey_step(self):
        query = self.nn_obj.rand_query()
        neighbors, intruder = self.nn_obj.rand_neighbors(query, 2, 0.7)

        q = None
        if any(neighbors) and intruder:
            q = CoherenceQuestion(query, neighbors[0], neighbors[1], intruder)
            q.index()
        return q

    def start_survey(self):
        try:
            q_index = 0
            while True:
                q = self.survey_step()
                if q:
                    print()
                    print(f"Question {q_index}.")
                    print(f'-' * 80)
                    print(q)
                    answer = get_answer()
                    if answer == 'n':
                        continue
                    else:
                        q.answer = answer
                        self.questions.append(q)
                    q_index += 1

        except KeyboardInterrupt:
            if len(self.questions):
                result = [q.is_answer_correct() for q in self.questions]
                acc = round(sum(result) * 100 / len(result), 2)
                print()
                print(f"Intruision detection rate: {acc}")
                print()
