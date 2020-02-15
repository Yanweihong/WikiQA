import codecs
import logging
import pickle
from collections import Counter

import jieba
import jieba.analyse

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

# 让jieba闭嘴！
jieba.setLogLevel(logging.INFO)

# 全局停用词
stopwords = []
st = codecs.open(r'D:\workspace\pycharm\myQA\stopwords.txt', 'r', 'utf-8')
for line in st:
    line = line.strip()
    stopwords.append(line)

# 取_dict中value最大的N个pairs
def get_order_dict_N(_dict, N):
    result = Counter(_dict).most_common(N)
    d = {}
    for k, v in result:
        d[k] = v
    return d


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

    def __init__(self, tfidf_path=None, query=''):
        tfidf_path = r'D:\workspace\pycharm\myQA\tfidf.pkl'

        logger.info("TFIDF model loading……")

        self.docs_keywords = pickle.load(codecs.open(tfidf_path, 'rb'))

        logger.info("TFIDF model loaded")
        self.query = query

    def closest_docs(self, query):
        query_keywords = query_process(query)

        closest_doc = {}

        for doc_id, doc_keywords in self.docs_keywords.items():
            weight = 0
            for keyword in query_keywords:
                for k in doc_keywords:
                    if keyword == k[0]:
                        weight = weight + k[1]
            if weight != 0:
                closest_doc[doc_id] = weight
        r = get_order_dict_N(closest_doc,5)
        return r


if __name__ == '__main__':
    ranker = TfidfDocRanker()
    print(ranker.closest_docs(query='中国的首都在哪'))
