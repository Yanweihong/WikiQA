from flask import Flask, render_template, jsonify, request

from wikiqa.retriever import sklearn_tfidf_ranker

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['post'])
def upload():
    query = request.form.get('query')
    print(query)
    ranker = sklearn_tfidf_ranker.TfidfDocRanker(query=query)
    ids, scores = ranker.closest_docs()
    print(ids, scores)

    return jsonify({'ids': ids, "scores": scores})


if __name__ == '__main__':
    app.run()
