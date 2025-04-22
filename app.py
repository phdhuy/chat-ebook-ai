from flask import Flask
from flasgger import Swagger
from flask_cors import CORS
import nltk
from utils.logging import setup_logging
from models.nlp_models import initialize_models
from services.elasticsearch_service import connect_elasticsearch
from routes.api import register_routes
from config import ES_INDEX

# Download NLTK data
nltk.download('punkt_tab')

# Initialize Flask app
app = Flask(__name__)
swagger = Swagger(app)
CORS(app)

# Set up logging
logger = setup_logging()

# Initialize models and Elasticsearch
embedder, model = initialize_models()
es = connect_elasticsearch()

# Register API routes
register_routes(app, es, embedder, model, ES_INDEX)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)