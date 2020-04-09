from flask import Flask, render_template, request

from logic import *

import os

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

print('Loading data.')
data = load_data()

print('Loading index.')
index = load_index()
print(type(index))

#print('Loading DistilmBERT.')
#tokenizer, model = load_bert()
print('Loading MUSE.')
model = load_muse()
#print('TEST PREDICTION:', model('Test prediction'))

print('Done.')

application = Flask(__name__)
#application.debug = True

@application.route('/', methods=['GET', 'POST'])
def home():
    papers = []

    if request.method == 'POST':

        input_text = request.form.get('query')
        if input_text:
            #query_embedding = get_query(input_text, tokenizer, model)
            query_embedding = get_muse_query(input_text, model)

            if request.form.get('submit') == 'top10': n_results = 10
            else: n_results = 100 #2000

            result_indices = get_results(query_embedding, index, n_results)
            papers = []
            keys = ['index', 'title', 'cord_uid', 'doi', 'source_x', 'pmcid', 'pubmed_id', 'license', 'abstract', 'publish_time', 'authors', 'journal', 'url']
            invalidAbs = False
            for doc_index in result_indices:
                result = data.iloc[doc_index]
                #print(result['abstract'])
                
                result_dict = {}
                for key in keys:
                    result_dict[key] = result[key]
                papers.append(result_dict)

    return render_template('index.html', papers=papers)

# run the app.
if __name__ == "__main__":

    application.run()