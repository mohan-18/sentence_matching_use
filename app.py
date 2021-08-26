from scipy.spatial import distance
from flask import Flask, request, jsonify
import ast
import pickle
import tensorflow as tf
import tensorflow_hub as hub
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# embeddings = embed(["hi", "bye"])
# print(1-distance.cosine(embeddings[0], embeddings[1]))

app = Flask(__name__)

# filename = 'nlp.pkl'
# nlp = pickle.load(open(filename, 'rb'))


@app.route('/')
def hello():
    return "Hello World"


@app.route('/similar', methods=['POST'])
def similar():
    data = request.data
    dict_str = data.decode("UTF-8")
    my_data = ast.literal_eval(dict_str)
    list = []

    for heading in my_data['article_headings']:
        # print(question, my_data['ques'])
        embeddings = embed([heading, my_data['customer_question']])
        list.append(
            [1-distance.cosine(embeddings[0], embeddings[1]), heading])
        list.sort(key=lambda x: x[0], reverse=True)
        list = list[0:3]
    return jsonify(list)


if __name__ == '__main__':
    app.run(debug=True)
