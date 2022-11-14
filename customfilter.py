import opusfilter


class NonAlphaFilter(opusfilter.FilterABC):

    def __init__(self, threshold=0.2, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)

    @staticmethod
    def alphas_to_nonalphas(chars):
        if not chars:
            return 0
        alphas = sum(1 for char in chars if char.isalpha())
        return alphas / len(chars)

    def score(self, pairs):
        for sent1, sent2 in pairs:
            yield {'src': self.alphas_to_nonalphas(sent1),
                   'tgt': self.alphas_to_nonalphas(sent2)}

    def accept(self, score):
        ratio1, ratio2 = score['src'], score['tgt']
        return ratio1 >= self.threshold and ratio2 >= self.threshold