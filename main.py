import MeCab
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

class TextExtractor(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, rows):
        return rows[:, 0]

class OtherFeaturesExtractor(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, rows):
        return rows[:, 1:].astype('float')

class WordDividor:
    INDEX_CATEGORY = 0
    INDEX_ROOT_FORM = 6
    TARGET_CATEGORIES = ["名詞", " 動詞",  "形容詞", "副詞", "連体詞", "感動詞"]

    def __init__(self, dictionary="mecabrc"):
        self.dictionary = dictionary
        self.tagger = MeCab.Tagger(self.dictionary)

    def extract_words(self, text):
        if not text:
            return []

        words = []

        node = self.tagger.parseToNode(text)
        while node:
            features = node.feature.split(',')

            if features[self.INDEX_CATEGORY] in self.TARGET_CATEGORIES:
                if features[self.INDEX_ROOT_FORM] == "*":
                    words.append(node.surface)
                else:
                    # prefer root form
                    words.append(features[self.INDEX_ROOT_FORM])

            node = node.next

        return words


if __name__ == '__main__':
    train_data = np.array([
        ['蛙の子は蛙', 0.5, 0.5],
        ['親の心子知らず', 0.2, 0.7]
    ])

    train_labels = [1, 2]

    test_data = np.array([
        ['鬼の居ぬ間に洗濯', 0.1, 0.8],
        ['取らぬ狸の皮算用', 0.4, 0.3]
    ])

    wd = WordDividor()
    cv = CountVectorizer(analyzer=wd.extract_words)

    clf1 = Pipeline([
        ('count_vector', cv),
        ('tfidf', TfidfTransformer()),
        ('classifier', SGDClassifier(loss='hinge', random_state=42))
    ])

    clf1.fit(train_data[:, 0], train_labels)
    print("text only:")
    print(clf1.predict(test_data[:, 0]))

    clf2 = Pipeline([
        ('features', FeatureUnion([
            ('text', Pipeline([
                ('content', TextExtractor()),
                ('count_vector', cv),
                ('tfidf', TfidfTransformer())
            ])),
            ('other_features', OtherFeaturesExtractor()),
        ])),
        ('classifier', SGDClassifier(loss='hinge', random_state=42))
    ])

    clf2.fit(train_data, train_labels)
    print("feature union:")
    print(clf2.predict(test_data))
