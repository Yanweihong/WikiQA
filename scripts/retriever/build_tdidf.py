import codecs
import os
import pickle
from multiprocessing import Pool

import jieba
import jieba.analyse
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from wikiqa.retriever.doc_db import DocDB

# ------------------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------------------

PROCESS_DB = DocDB(db_path=r'D:\workspace\pycharm\myQA\v2.db')

DOC2IDX = {}

# 全局停用词
stopwords = []
st = codecs.open(r'D:\workspace\pycharm\myQA\stopwords.txt', 'r', 'utf-8')
for line in st:
    line = line.strip()
    stopwords.append(line)


def fetch_text(doc_id):
    global PROCESS_DB
    return PROCESS_DB.get_doc_text((doc_id,))


def fetch_ids():
    global PROCESS_DB
    return PROCESS_DB.get_doc_ids()


# ------------------------------------------------------------------------------
# jieba.analyse提取TopK关键词及其tfidf权重
# ------------------------------------------------------------------------------

def fetch_tfidf(doc_id):
    article = fetch_text(doc_id)
    keywords = jieba.analyse.extract_tags(article, topK=15, withWeight=True)
    return doc_id, keywords


def store_model(model_path):
    keywords = {}
    doc_ids = fetch_ids()
    with tqdm(total=954908) as pbar:
        with Pool() as workers:
            for doc_id, doc_keywords in tqdm(workers.imap_unordered(fetch_tfidf, doc_ids)):
                keywords[doc_id] = doc_keywords
                pbar.update()
    f = codecs.open(model_path, 'wb')
    pickle.dump(keywords, f)


def load_model(model_path):
    f = codecs.open(model_path, 'rb')
    return pickle.load(f)


# ------------------------------------------------------------------------------
# sklearn.feature_extraction
# ------------------------------------------------------------------------------

def tokensize(doc_id):
    article = fetch_text(doc_id)
    return doc_id, " ".join(jieba.cut(article))


def store_tfidf_model(model_path):
    global DOC2IDX
    global stopwords
    corpus = []
    doc_ids = fetch_ids()
    with tqdm(total=954908) as pbar:
        with Pool() as workers:
            index = -1
            for doc_id, d in tqdm(workers.imap_unordered(tokensize, doc_ids)):
                index += 1
                DOC2IDX[index] = doc_id
                corpus.append(d)
                pbar.update()

    # store corpus
    corpus_path = r'D:\workspace\pycharm\myQA\model\corpus.pkl'
    if not os.path.exists(corpus_path):
        f1 = codecs.open(corpus_path, 'wb')
        pickle.dump(corpus, f1)
    else:
        corpus = pickle.load(codecs.open(corpus_path, 'rb'))

    # store vectorizer
    tfidf_model = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=stopwords, ngram_range=(1, 2)).fit(corpus)

    f2 = codecs.open(model_path, 'wb')
    pickle.dump(tfidf_model, f2)

    # store matrix
    f3 = codecs.open(r'D:\workspace\pycharm\myQA\model\tfidf_matrix.pkl', 'wb')
    pickle.dump(tfidf_model.transform(corpus), f3)

    f4 = codecs.open(r'D:\workspace\pycharm\myQA\model\doc2idx.pkl', 'wb')
    pickle.dump(DOC2IDX, f4)


# ------------------------------------------------------------------------------
# elasticsearch_ranker
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    store_tfidf_model(r'D:\workspace\pycharm\myQA\model\sklearn_tfidf.pkl')
