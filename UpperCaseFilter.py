import opusfilter


class UppercaseFilter(opusfilter.FilterABC):

    def __init__(self, threshold=0.5, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)

    def uppercase_ratio(self, sentence):
        length = len(sentence)
        if length > 0:
            return sum(1 for char in sentence if char.isupper()) / length
        return 0

    def score(self, pairs):
        for pair in pairs:
            yield [self.uppercase_ratio(sentence) for sentence in pair]

    def accept(self, score):
        return all(ratio < self.threshold for ratio in score)
