from data_process import data_process
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

adata, data_after_stop, lables = data_process()
data_tr, data_te, labels_tr, labels_te = train_test_split(adata, lables, test_size=0.2)

countVectorizer = CountVectorizer()
data_tr = countVectorizer.fit_transform(data_tr)
X_tr = TfidfTransformer().fit_transform(data_tr.toarray()).toarray()

data_te = CountVectorizer(vocabulary=countVectorizer.vocabulary_).fit_transform(data_te)
X_te = TfidfTransformer().fit_transform(data_te.toarray()).toarray()

model = GaussianNB()
model.fit(X_tr, labels_tr)
model.score(X_te, labels_te)



