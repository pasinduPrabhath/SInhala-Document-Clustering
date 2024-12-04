from flask import Flask, request, jsonify
from sinhala_ner.ner_detector import SinhalaNERDetector
from sinhala_ner.document_clustering import DocumentClusterer
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Initialize components with validation
API_KEY = os.getenv('PERPLEXITY_API_KEY')
if not API_KEY:
    raise ValueError("Perplexity API key is missing")

WORD2VEC_MODEL_PATH = 'word2Vec/sinhala_word2vec.model'
SPACY_MODEL_PATH = 'spacy_model'

# Initialize detectors
ner_detector = SinhalaNERDetector(API_KEY)
document_clusterer = DocumentClusterer(
    word2vec_model_path=WORD2VEC_MODEL_PATH,
    api_key=API_KEY
)

@app.route('/detect_ner', methods=['POST'])
def detect_ner():
    try:
        sentences = request.json.get('sentences', [])
        if not sentences:
            return jsonify({"error": "No sentences provided"}), 400
        
        results = ner_detector.detect_entities(sentences)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/detect_and_cluster', methods=['POST'])
def detect_and_cluster():
    try:
        sentences = request.json.get('sentences', [])
        if not sentences:
            return jsonify({"error": "No sentences provided"}), 400
        
        # Perform NER detection
        ner_results = ner_detector.detect_entities(sentences)
        
        # Perform clustering
        clustering_results = document_clusterer.cluster_documents(ner_results)
        
        return jsonify(clustering_results)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask application...")
    print(f"API Key present: {'Yes' if API_KEY else 'No'}")
    print(f"Word2Vec model path: {WORD2VEC_MODEL_PATH}")
    print(f"SpaCy model path: {SPACY_MODEL_PATH}")
    app.run(debug=True, port=5000)