# Copyright (C) 2013 Wesley Baugh
"""Tools for text classification.
Extracted from the [infer](https://github.com/bwbaugh/infer) library.
"""
from __future__ import division

import math
from collections import defaultdict, namedtuple, Counter
from fractions import Fraction


class MultinomialNB(object):
    """Multinomial Naive Bayes for text classification.
    Attributes:
        exact: Boolean indicating if exact probabilities should be
            returned as a `Fraction`. Otherwise, speed up computations
            but only return probabilities as a `float`. (default False)
        laplace: Smoothing parameter >= 0. (default 1)
        top_features: Number indicating the top-k most common features
            to use during classification, sorted by the frequency the
            feature has been seen (a count is kept for each label). This
            is a form of feature selection because any feature that has
            a frequency less than any of the top-k most common features
            is ignored during classification. This value must be set
            before any training of the classifier. (default None)
    Properties:
        labels: Set of all class labels.
        vocabulary: Set of vocabulary across all class labels.
    """
    Prediction = namedtuple('Prediction', 'label confidence')

    def __init__(self, *documents):
        """Create a new Multinomial Naive Bayes classifier.
        Args:
            documents: Optional list of document-label pairs for training.
        """
        self.exact = False
        self.laplace = 1
        self.top_features = None
        # Dictionary of sets of vocabulary by label.
        self._label_vocab = defaultdict(set)
        # Dictionary of times a label has been seen.
        self._label_count = Counter()
        # Dictionary of number of feature seen in all documents by label.
        self._label_length = Counter()
        # Dictionary of times a feature has been seen by label.
        self._label_feature_count = defaultdict(Counter)
        # Size of vocabulary across all class labels.
        self._vocab_size = 0
        if documents:
            self.train(*documents)

    @property
    def labels(self):
        """Set of all class labels.
        Returns:
            Example: set(['positive', 'negative'])
        """
        return set(label for label in self._label_count)

    @property
    def vocabulary(self):
        """Set of vocabulary (features) seen in any class label."""
        label_vocab = [self._label_vocab[x] for x in self._label_vocab]
        return set().union(*label_vocab)

    def train(self, *documents):
        """Train the classifier on a document-label pair(s).
        Args:
            documents: Tuple of (document, label) pair(s). Documents
                must be a collection of features. The label can be any
                hashable object, though is usually a string.
        """
        for document, label in documents:
            # Python 3: isinstance(document, str)
            if isinstance(document, basestring):
                raise TypeError('Documents must be a collection of features')
            self._label_count[label] += 1
            for feature in document:
                # Check if the feature hasn't been seen before for any label.
                if not any(feature in self._label_vocab[x] for x in self.labels):
                    self._vocab_size += 1
                self._label_vocab[label].add(feature)
                self._label_feature_count[label][feature] += 1
                self._label_length[label] += 1
                if self.top_features:
                    if not hasattr(self, '_most_common'):
                        x = lambda: MostCommon(self.top_features)
                        self._most_common = defaultdict(x)
                    y = self._label_feature_count[label][feature]
                    self._most_common[label][feature] = y

    def prior(self, label):
        """Prior probability of a label.
        Args:
            label: The target class label.
            self.exact
        Returns:
            The number of training instances that had the target
            `label`, divided by the total number of training instances.
        """
        if label not in self.labels:
            raise KeyError(label)
        total = sum(self._label_count.values())
        if self.exact:
            return Fraction(self._label_count[label], total)
        else:
            return self._label_count[label] / total

    def conditional(self, feature, label):
        """Conditional probability for a feature given a label.
        Args:
            feature: The target feature.
            label: The target class label.
            self.laplace
            self.exact
        Returns:
            The number of times the feature has been present across all
            training documents for the `label`, divided by the sum of
            the length of every training document for the `label`.
        """
        # Note we use [Laplace smoothing][laplace].
        # [laplace]: https://en.wikipedia.org/wiki/Additive_smoothing
        if label not in self.labels:
            raise KeyError(label)

        # Times feature seen across all documents in a label.
        numer = self.laplace
        # Avoid creating an entry if the term has never been seen
        if feature in self._label_feature_count[label]:
            numer += self._label_feature_count[label][feature]
        denom = self._label_length[label] + (self._vocab_size * self.laplace)
        if self.exact:
            return Fraction(numer, denom)
        else:
            return numer / denom

    def _score(self, document, label):
        """Multinomial raw score of a document given a label.
        Args:
            document: Collection of features.
            label: The target class label.
            self.exact
        Returns:
            The multinomial raw score of the `document` given the
            `label`. In order to turn the raw score into a confidence
            value, this value should be divided by the sum of the raw
            scores across all class labels.
        """
        if isinstance(document, basestring):
            raise TypeError('Documents must be a list of features')

        if self.exact:
            score = self.prior(label)
        else:
            score = math.log(self.prior(label))

        for feature in document:
            # Feature selection by only considering the top-k
            # most common features (a form of dictionary trimming).
            if self.top_features and feature not in self._most_common[label]:
                continue
            conditional = self.conditional(feature, label)
            if self.exact:
                score *= conditional
            else:
                score += math.log(conditional)

        return score

    def _compute_scores(self, document):
        """Compute the multinomial score of a document for all labels.
        Args:
            document: Collection of features.
        Returns:
            A dict mapping class labels to the multinomial raw score
            for the `document` given the label.
        """
        return {x: self._score(document, x) for x in self.labels}

    def prob_all(self, document):
        """Probability of a document for all labels.
        Args:
            document: Collection of features.
            self.exact
        Returns:
            A dict mapping class labels to the confidence value that the
            `document` belongs to the label.
        """
        score = self._compute_scores(document)
        if not self.exact:
            # If the log-likelihood is too small, when we convert back
            # using `math.exp`, the result will round to zero.
            normalize = max(score.itervalues())
            assert normalize <= 0, normalize
            score = {x: math.exp(score[x] - normalize) for x in score}
        total = sum(score[x] for x in score)
        assert total > 0, (total, score, normalize)
        if self.exact:
            return {label: Fraction(score[label], total) for label in
                    self.labels}
        else:
            return {label: score[label] / total for label in self.labels}

    def prob(self, document, label):
        """Probability of a document given a label.
        Args:
            document: Collection of features.
            label: The target class label.
        Returns:
            The confidence value that the `document` belongs to `label`.
        """
        prob = self.prob_all(document)[label]
        return prob

    def classify(self, document):
        """Get the most confident class label for a document.
        Args:
            document: Collection of features.
        Returns:
            A namedtuple representing the most confident class `label`
            and the value of the `confidence` in the label. For example:
            As tuple:
                ('positive', 0.85)
            As namedtuple:
                Prediction(label='positive', confidence=0.85)
        """
        prob = self.prob_all(document)
        label = max(prob, key=prob.get)
        return self.Prediction(label, prob[label])
