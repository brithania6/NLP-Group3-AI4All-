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
