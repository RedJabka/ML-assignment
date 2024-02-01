import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

def validate_model(val_file, model_file, output_file, vec_file):
    # Загружаем валидационные данные
    val_data = pd.read_csv(val_file, sep='\t')

    # Загружаем модель
    model = pickle.load(open(model_file, 'rb'))

    # Загружаем векторизацию
    vectorizer = pickle.load(open(vec_file, 'rb'))
    X_val = vectorizer.transform(val_data['libs'])
    y_true = val_data['is_virus']

    # Предскажем на валидационной выборке
    y_pred = model.predict(X_val)

    # Оценка модели
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1][1]
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    # Запись оценки в файл
    with open(output_file, 'w') as f:
        f.write(f'True Positive: {tp}\n')
        f.write(f'False Positive: {fp}\n')
        f.write(f'False Negative: {fn}\n')
        f.write(f'True Negative: {tn}\n')
        f.write(f'Precision: {precision}\n')
        f.write(f'Accuracy: {accuracy}\n')
        f.write(f'Recall: {recall}\n')
        f.write(f'F1: {f1}\n')

if __name__ == "__main__":
    val_file = 'tsv_files/val.tsv'
    model_file = 'trained_model.model'
    output_file = 'created_files/validation.txt'
    vec_file = 'vec.pickle'

    validate_model(val_file, model_file, output_file, vec_file)