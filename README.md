from sklearn.datasets import fetch_20newsgroups

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB



def nb_news():

    news = fetch_20newsgroups(subset="all")
    x_train,x_test,y_train, y_test = train_test_split(news.data, news.target)
    transfer =  TfidfVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    estimator = MultinomialNB()
    estimator.fit(x_train, y_train)

    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接比对真实值和预测值:\n", y_test == y_predict)

    score = estimator.score(x_test, y_test)
    print("准确率为:\n", score)

    return None

if __name__ == "__main__":

    nb_news()
