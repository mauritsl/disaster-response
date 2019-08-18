import sys
import pandas as pd
import numpy as np
import nltk
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

nltk.download('punkt')
nltk.download('wordnet')

def load_data(database_filepath):
    """Load DataFrame from SQLite database file"""

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('message', engine, index_col='id')
    X_columns = ['message', 'original', 'genre']
    Y_columns = list(filter(lambda item : item not in X_columns, df.columns.values))
    X = df.message
    Y = df[Y_columns]

    return X, Y, Y_columns


def tokenize(text):
    """Split input text in tokens"""

    lemmatizer = WordNetLemmatizer()
    return list(map(lambda t : lemmatizer.lemmatize(t).lower().strip(), word_tokenize(text)))


def build_model():
    """Build ML pipeline"""

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier(random_state=42)))
    ])
    parameters = {
        'clf__estimator__n_estimators': [20],
        'clf__estimator__learning_rate': [0.5, 1.0, 1.5]
    }
    return GridSearchCV(pipeline, parameters)


def evaluate_model(model, X_test, Y_test, category_names):
    """Print model's performance indicators"""

    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred)
    y_pred.columns = Y_test.columns
    y_pred.index = Y_test.index

    accuracy = 0
    for col in y_pred:
        print('{:25}: F1: {:01.2f} Precision: {:01.2f} Recall: {:01.2f}'.format(
            col,
            f1_score(Y_test[col] == 1, y_pred[col] == 1),
            precision_score(Y_test[col] == 1, y_pred[col] == 1),
            recall_score(Y_test[col] == 1, y_pred[col] == 1)
        ))
        accuracy += (Y_test[col] == y_pred[col]).mean()
    accuracy /= len(y_pred.columns)
    print('Overall accuracy: {:04.2}'.format(accuracy))


def save_model(model, model_filepath):
    """Save model to disk"""

    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """Main application function"""

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
