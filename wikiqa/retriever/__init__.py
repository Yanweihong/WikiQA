import os

import tools

root_path = tools.getRootPath()

DEFAULTS = {
    'model_path': os.path.join(root_path, 'model'),
    'tfidf_path': os.path.join(root_path, 'model/sklearn_tfidf.pkl'),
    'tfidf_matrix_path': os.path.join(root_path, 'model/tfidf_matrix.pkl'),
    'doc_dict_path': os.path.join(root_path, 'model/doc2idx.pkl'),
    'db_path': os.path.join(root_path, 'v2.db')
}
