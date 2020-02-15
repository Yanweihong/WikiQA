import codecs
import logging
import pickle

import jieba
import jieba.analyse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from wikiqa.retriever.doc_db import DocDB
from wikiqa.retriever import DEFAULTS

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

# 让jieba闭嘴！
jieba.setLogLevel(logging.INFO)

PROCESS_DB = DocDB(db_path=DEFAULTS['db_path'])

def fetch_text(doc_id):
    global PROCESS_DB
    return PROCESS_DB.get_doc_text((doc_id,))

# 全局停用词
stopwords = []
st = codecs.open(r'D:\workspace\pycharm\myQA\stopwords.txt', 'r', 'utf-8')
for line in st:
    line = line.strip()
    stopwords.append(line)

def query_process(query):
    """
    tokenize query from str to word list, then remove stopwords
    """
    query_words = [w for w in jieba.cut_for_search(query)]
    for query_word in query_words:
        if query_word in stopwords:
            query_words.remove(query_word)
    return query_words


class TfidfDocRanker(object):

    def __init__(self, tfidf_path=None, query='', k=5, doc_dict=None):
        tfidf_path = DEFAULTS['tfidf_path']
        tfidf_matrix_path = DEFAULTS['tfidf_matrix_path']
        doc_dict_path = DEFAULTS['doc_dict_path']
        logger.info("TFIDF model loading……")
        self.tfidf_vectorizer = pickle.load(codecs.open(tfidf_path, 'rb'))
        self.tfidf_matrix = pickle.load(codecs.open(tfidf_matrix_path, 'rb'))
        self.doc_dict = pickle.load(codecs.open(doc_dict_path, 'rb'))
        logger.info("TFIDF model loaded")
        self.query = " ".join(jieba.cut(query))

    def get_doc_id(self, doc_index):
        """Convert doc_index --> doc_id"""
        return self.doc_dict[1][doc_index]

    def closest_docs(self, k=5):
        query = self.query
        vectorizer = self.tfidf_vectorizer
        docs_tfidf = self.tfidf_matrix
        query_tfidf = vectorizer.transform([query])
        cosine_sim = cosine_similarity(query_tfidf, docs_tfidf).flatten()

        ind = np.argpartition(cosine_sim, -k)[-k:]
        o_sort = ind[np.argsort(-cosine_sim[ind])]
        doc_scores = cosine_sim[o_sort]
        doc_ids = []
        for i in o_sort:
            doc_ids.append(self.doc_dict[i])
        # return o_sort.tolist(), doc_scores.tolist()
        logger.info("Rank Finished")
        return doc_ids, doc_scores.tolist()


# if __name__ == '__main__':
#     query = '伦敦是哪个国家的城市'
#     ranker = TfidfDocRanker(query=query)
#     ids, scores = ranker.closest_docs()
#     print(ids)
#     print(scores)