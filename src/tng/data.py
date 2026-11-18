from functools import lru_cache

import torch
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, TensorDataset


@lru_cache(maxsize=1)
def english_stopwords():
    return frozenset(stopwords.words("english"))


def preprocess_text(text, remove_stopwords=False, normalization="none"):
    tokens = [token for token in word_tokenize(text.lower()) if token.isalpha()]
    if remove_stopwords:
        excluded = english_stopwords()
        tokens = [token for token in tokens if token not in excluded]
    if normalization == "stem":
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
    elif normalization == "lemmatize":
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(tokens)


def vectorize_data(train_texts, evaluation_texts, ngram_range, max_features, min_df):
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range, max_features=max_features, min_df=min_df
    )
    train_features = vectorizer.fit_transform(train_texts)
    evaluation_features = vectorizer.transform(evaluation_texts)
    return train_features, evaluation_features, vectorizer


def make_loader(features, labels, batch_size, *, shuffle=False, seed=42):
    dataset = TensorDataset(
        torch.from_numpy(features.astype("float32", copy=False).toarray()),
        torch.tensor(labels, dtype=torch.long),
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
    )
