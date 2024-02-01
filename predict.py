import pandas as pd
import pickle

def predict_model(test_file, model_file, prediction_file, explain_file, vec_file):
    # Загружаем проверочные данные
    test_data = pd.read_csv(test_file, sep='\t')

    # Загружаем модель
    model = pickle.load(open(model_file, 'rb'))

    # Загружаем векторизацию
    vectorizer = pickle.load(open(vec_file, 'rb'))
    X_test = vectorizer.transform(test_data['libs'])

    # Предскажем на проверочной выборке
    y_test_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)

    # Запись предсказание в файл
    with open(prediction_file, 'w') as f_pred:
        f_pred.write('prediction\n')
        for prediction in y_test_pred:
            f_pred.write(f'{prediction}\n')

    # Запись объяснений в файл
    with open(explain_file, 'w') as f_explain:
        for i, prediction in enumerate(y_test_pred):
            if prediction == 1:  # Если модель считает файл зловредным
                f_explain.write(f'Файлы: {test_data["libs"][i]}\n')
                f_explain.write(f'Причина: Высокая вероятность быть зловредным - {proba[i][1]:.4f}\n\n')

if __name__ == "__main__":
    test_file = 'tsv_files/test.tsv'
    model_file = 'trained_model.model'
    prediction_file = 'created_files/prediction.txt'
    explain_file = 'created_files/explain.txt'
    vec_file = 'vec.pickle'

    predict_model(test_file, model_file, prediction_file, explain_file, vec_file)