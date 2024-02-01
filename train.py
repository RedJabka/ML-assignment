import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

def train_model(train_file, model_file, vec_file):
    # Загружаем тренировочные данные
    train_data = pd.read_csv(train_file, sep='\t')

    # Предобработаем данные
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_data['libs'])
    y_train = train_data['is_virus']

    # Обучаем модель
    model = MultinomialNB() # Выбирал между наивным Байесовский классификатором и Градиентным спуском. Байес дал результаты лучше
    model.fit(X_train, y_train)

    # Сохраняем векторизацию
    pickle.dump(vectorizer, open(vec_file, 'wb'))

    # Сохраненяем модель
    pickle.dump(model, open(model_file, 'wb'))

if __name__ == "__main__":
    train_file = 'tsv_files/train.tsv'
    model_file = 'trained_model.model'
    vec_file = 'vec.pickle'

    train_model(train_file, model_file, vec_file)