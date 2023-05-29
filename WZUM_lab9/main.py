import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def example():
    text = "I'm having a wonderful time at WZUM laboratories! WZUM changed my life!!!"
    print(text.split())

    tokenizer = CountVectorizer().build_tokenizer()
    print(tokenizer(text))

    count_vect = CountVectorizer()
    counts = count_vect.fit_transform([text])
    print(counts)

    print('-'*30)
    tf = TfidfTransformer(use_idf=False).fit_transform(counts)
    print(tf)


# TODO 1 - 5 -----------------------------------------------------------------------------------------------------------

def ex_1_5():
    categories = ['sci.space', 'rec.sport.hockey', 'comp.graphics', 'sci.med']
    twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

    # print(twenty_train.data[0])

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(twenty_train.data)
    # print(X_train_counts)

    # print(count_vect.vocabulary_.get('data'))
    # print(count_vect.transform(['data']))
    # print(count_vect.vocabulary_.get('algorithm'))
    # print(count_vect.transform(['algorithm']))
    # print(count_vect.vocabulary_.get('laboratory'))
    # print(count_vect.transform(['laboratory']))
    # print(count_vect.vocabulary_.get('WZUM'))
    # print(count_vect.transform(['WZUM']))

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    # print(X_train_tfidf)

    clf = MultinomialNB()
    clf.fit(X_train_tfidf, twenty_train.target)

    docs_new = ['There was a new planet discovered',
                'There was a new organ discovered',
                'OpenGL on the GPU is fast']

    X_new_counts = count_vect.transform(docs_new)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    predicted = clf.predict(X_new_tfidf)

    for doc, category in zip(docs_new, predicted):
        print('%r => %s' % (doc, twenty_train.target_names[category]))


def ex_6_7():
    categories = ['sci.space', 'rec.sport.hockey', 'comp.graphics', 'sci.med']
    twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
    twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

    text_clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB()),
    ])
    text_clf.fit(twenty_train.data, twenty_train.target)

    predicted = text_clf.predict(twenty_test.data)
    print(np.mean(predicted == twenty_test.target))


def ex_8_9():
    categories = ['sci.space', 'rec.sport.hockey', 'comp.graphics', 'sci.med']
    twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
    twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

    text_clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', SVC()),
    ])
    text_clf.fit(twenty_train.data, twenty_train.target)

    predicted = text_clf.predict(twenty_test.data)
    print(np.mean(predicted == twenty_test.target))

    cm = confusion_matrix(twenty_test.target, predicted)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()


if __name__ == '__main__':
    ex_8_9()
