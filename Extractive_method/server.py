from flask import Flask, jsonify, request
from flask_cors import CORS
from phobert_extractive_text_summ import phoBertExtractiveModel

DEBUG = True

app = Flask(__name__)
app.config.from_object(__name__)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

model = phoBertExtractiveModel()

# sanity check route
@app.route('/raw', methods=['GET'])
def rawtextsummerizer():
    query = request.args.get("text")
    summary = model.get_summarization(query)
    return jsonify({
        'answer': summary
    })

@app.route('/url', methods=['GET'])
def urltextsummerizer():
    query = request.args.get("text")
    print(query)
    return jsonify({
        'answer': 'hello from url'
    })

if __name__ == '__main__':
    app.run()