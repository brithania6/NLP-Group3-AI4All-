# NLP-Group3-AI4All-
NLP modeling building 
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch
from torchvision import models,transforms
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from pickle import dump
from torch.utils.data import Dataset,DataLoader
from collections import Counter
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
device = torch.device('cpu')
data = pd.read_csv("../input/jobposts/data job posts.csv")
data

# %%
import numpy as np

mock_data = {
    "Job Title": ["Software Engineer", "Data Scientist", "Product Manager", None, "UX Designer"],
    "Company": ["Google", "Amazon", "Meta", "Apple", None],
    "Location": ["New York, NY", "Seattle, WA", None, "San Francisco, CA", "Austin, TX"],
    "Posted Date": ["2024-10-15", "2024-11-01", "InvalidDate", "2024-10-25", "2024-11-10"],
    "Skills": ["Python, Machine Learning", None, "Agile, Product Strategy", "Swift, Objective-C", "Figma, UX Research"],
    "Salary": ["120000-140000", None, "130000-150000", "115000-135000", "105000-125000"]
}

mock_df = pd.DataFrame(mock_data)

mock_df.info(), mock_df.head()


# %%
data.RequiredQual

# %%
data.columns

# %%
data.JobDescription

# %%
data.JobRequirment

# %%
df = data[["RequiredQual","JobDescription","JobRequirment","Title"]].dropna()
df

# %%
classes = df['Title'].value_counts()[:20]
keys = classes.keys().to_list()

df = df[df['Title'].isin(keys)]
df['Title'].value_counts()

# %%
def chane_titles(x):
    x = x.strip()
    if x == 'Senior Java Developer':
        return 'Java Developer'
    elif x == 'Senior Software Engineer':
        return 'Software Engineer'
    elif x == 'Senior QA Engineer':
        return 'Software QA Engineer'
    elif x == 'Senior Software Developer':
        return 'Senior Web Developer'
    elif x =='Senior PHP Developer':
        return 'PHP Developer'
    elif x == 'Senior .NET Developer':
        return '.NET Developer'
    elif x == 'Senior Web Developer':
        return 'Web Developer'
    elif x == 'Database Administrator':
        return 'Database Admin/Dev'
    elif x == 'Database Developer':
        return 'Database Admin/Dev'

    else:
        return x
        
    
df['Title'] = df['Title'].apply(chane_titles)
df['Title'].value_counts()

# %%
df["Combined"] = df.RequiredQual + df.JobDescription + df.JobRequirment
df.Combined = df.Combined.apply(lambda x: x.replace("\r\n"," "))
df

# %%
df.iloc[0,4]

# %%
df.to_csv("Modified.csv",index=False)

# %%
class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

# %%
def build_vocab(df, threshold=3):
    """Build a simple vocabulary wrapper."""
    counter = Counter()
    
    for i in range(len(df)):
        caption = df.iloc[i,4]
        tokens = nltk.tokenize.word_tokenize(str(caption))
        counter.update(tokens)

        if (i+1) % 1000 == 0:
                print("[{}/{}] Tokenized the sentences.".format(i+1, len(df)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

# %%
import pandas as pd
from pickle import dump

# Example DataFrame
data = {'text_column': ['this is a test', 'another example']}
df = pd.DataFrame(data)

# Define the build_vocab function
def build_vocab(dataframe):
    vocab = set()
    for text in dataframe['text_column']:  # Replace 'text_column' with your column name
        vocab.update(text.split())
    return vocab

# Build vocabulary and save it
v = build_vocab(df)
dump(v, open('vocab.pkl', 'wb'))
print(len(v))  # Output the length of the vocabulary

# %%
from sklearn.preprocessing import LabelEncoder  # Import LabelEncoder

# Initialize the LabelEncoder
le = LabelEncoder()

# Example DataFrame
import pandas as pd
data = {'Title': ['Engineer', 'Doctor', 'Artist', 'Engineer']}
df = pd.DataFrame(data)

# Apply LabelEncoder to the "Title" column
df["TitleUse"] = le.fit_transform(df.Title)
print(df)

# %%
df.iloc[:,5].nunique()

# %%
x = torch.Tensor(np.array(df.iloc[:,5]))
x

# %%
class Data(Dataset):
    def __init__(self,df,vocab):
        self.df = df
        self.vocab = vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        tokens = nltk.tokenize.word_tokenize(str(df.iloc[idx,4]))
        caption = []
        caption.append(self.vocab('<start>'))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('<end>'))
        return caption,x[idx]

def collate_fn(data):
    data.sort(key=lambda x: len(x[0]), reverse=True)
    captions,labels = zip(*data)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = torch.Tensor(cap[:end])        
    return targets.to(device), labels

# %%
X_train,X_test,y_train,y_test = train_test_split(df.Combined,df.Title,test_size = 0.15,random_state = 0)
train = Data(pd.DataFrame(X_train),v)
test = Data(pd.DataFrame(X_test),v)
dataloaderTrain = DataLoader(train,4,num_workers=0,collate_fn=collate_fn)
dataloaderTest = DataLoader(test,4,num_workers=0,collate_fn=collate_fn)

# %%
for i,j in dataloaderTrain:
    print(i)
    print(j)
    print(i.shape,torch.stack(j).shape)
    break

# %%
# train(model,dataloaderTrain,dataloaderTest,0.001,45)

# %%
# torch.save(model.state_dict(), "epoch8.pb")
#torch.save(model.state_dict(), "epoch8.pb")

# %%
# Example dummy data
import pandas as pd
import matplotlib.pyplot as plt

data = pd.DataFrame({
    'JobTitle': ['Engineer', 'Manager', 'Analyst', 'Developer', 'Engineer',
                 'Consultant', 'Engineer', 'Manager', 'Developer', 'Analyst']
})

job_counts = data['JobTitle'].value_counts().head(10)
job_counts.plot(kind='bar', color='skyblue', figsize=(10, 6))

plt.title('Top 10 Most Frequent Job Titles')
plt.xlabel('Job Titles')
plt.ylabel('Number of Postings')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
# Function to extract average salary from salary range
def extract_salary(salary_str):
    """Extracts average salary from a salary range string."""
    if pd.isna(salary_str):
        return np.nan
    elif '-' in salary_str:
        low, high = map(int, salary_str.split('-'))
        return (low + high) / 2  # Return the average salary
    return np.nan
# Function to preprocess the DataFrame
def preprocess_data(df):
    """Preprocesses the DataFrame by extracting salaries and encoding job titles."""
    # Extract average salary
    df['Salary'] = df['Salary'].apply(extract_salary)
    # Drop rows with NaN values in Salary
    df = df.dropna(subset=['Salary'])
    # One-hot encode categorical features (job titles)
    X = pd.get_dummies(df[['TitleUse']], drop_first=True)  # Use TitleUse as features
    y = df['Salary']  # Target variable
    return X, y
# Function to train the linear regression model
def train_linear_regression(X_train, y_train):
    """Trains a Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
# Function to evaluate the model
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


