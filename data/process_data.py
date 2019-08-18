import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load data from CSV files and combine them in a single DataFrame"""

    # Read CSV files.
    messages = pd.read_csv(messages_filepath, index_col='id')
    categories = pd.read_csv(categories_filepath, index_col='id')

    # Merge files.
    df = messages.merge(categories, on=['id'])

    # Split categories in separate columns.
    categories = categories.categories.str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = list(map(lambda item : item[0:-2], row))
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str.extract(r'(.)$').astype(int)
    df = df.drop(columns=['categories'])
    df = df.merge(categories, left_index=True, right_index=True)

    return df


def clean_data(df):
    """Cleanup DataFrame"""

    # Drop duplicates.
    df = df[df.duplicated() == False]

    return df


def save_data(df, database_filename):
    """Save DataFrame to disk"""

    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('message', engine, index=True)


def main():
    """Main application function"""

    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
