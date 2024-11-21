import pandas as pd
import numpy as np
import torch
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import Dataset, DataLoader
from pickle import dump
import matplotlib.pyplot as plt

# Download stopwords
nltk.download("stopwords")
nltk.download("punkt")

# Load the dataset
data = pd.read_csv("../input/jobposts/data job posts.csv")

# Preprocess the data
def extract_salary(salary_str):
    """Extracts average salary from a salary range string."""
    if pd.isna(salary_str):
        return np.nan
    elif '-' in salary_str:
        low, high = map(int, salary_str.split('-'))
        return (low + high) / 2  # Return the average salary
    return np.nan

def preprocess_data(df):
    """Preprocesses the DataFrame by extracting salaries and encoding job titles."""
    # Extract average salary
    df['Salary'] = df['Salary'].apply(extract_salary)
    # Drop rows with NaN values in Salary
    df = df.dropna(subset=['Salary'])
    # One-hot encode categorical features (job titles)
    df['Title'] = df['Title'].fillna('Unknown')  # Fill NaN values with 'Unknown'
    X = pd.get_dummies(df[['Title']], drop_first=True)  # Use Title as features
    y = df['Salary']  # Target variable
    return X, y

# Vocabulary class
class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        return self.word2idx.get(word, self.word2idx['<unk>'])

    def __len__(self):
        return len(self.word2idx)

# Build vocabulary
def build_vocab(df, threshold=3):
    """Build a simple vocabulary wrapper."""
    counter = Counter()
    
    for i in range(len(df)):
        caption = df.iloc[i, 4]
        tokens = word_tokenize(str(caption))
        counter.update(tokens)

        if (i + 1) % 1000 == 0:
            print("[{}/{}] Tokenized the sentences.".format(i + 1, len(df)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for word in words:
        vocab.add_word(word)
    return vocab

# Custom Dataset class
class JobDataset(Dataset):
    def __init__(self, df, vocab):
        self.df = df
        self.vocab = vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        tokens = word_tokenize(str(self.df.iloc[idx, 4]))
        caption = [self.vocab('<start>')] + [self.vocab(token) for token in tokens] + [self.vocab('<end>')]
        return caption, self.df.iloc[idx, 5]  # Assuming 5th column is the label

def collate_fn(data):
    data.sort(key=lambda x: len(x[0]), reverse=True)
    captions, labels = zip(*data)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = torch.Tensor(cap[:end])
    return targets, labels

# Train Linear Regression Model
def train_linear_regression(X_train, y_train):
    """Trains a Linear Regression model ."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    """Evaluates the trained model using Mean Squared Error and R^2 Score."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

# Main function to execute the workflow
def main():
    # Load the dataset
    data = pd.read_csv("../input/jobposts/data job posts.csv")
    
    # Preprocess the data
    X, y = preprocess_data(data)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)
    
    # Train the linear regression model
    model = train_linear_regression(X_train, y_train)
    
    # Evaluate the model
    mse, r2 = evaluate_model(model, X_test, y_test)
    
    # Print evaluation metrics
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')

if __name__ == "__main__":
    main()

# Visualization of job title frequency
def visualize_job_titles(data):
    job_counts = data['Job Title'].value_counts().head(10)
    job_counts.plot(kind='bar', color='skyblue', figsize=(10, 6))
    plt.title('Top 10 Most Frequent Job Titles')
    plt.xlabel('Job Titles')
    plt.ylabel('Number of Postings')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Call the visualization function
visualize_job_titles(data)


