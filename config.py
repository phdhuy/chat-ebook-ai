import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not set in environment variables")

ES_HOST = os.getenv('ES_HOST')
ES_API_KEY = os.getenv('ES_API_KEY')
ES_INDEX = os.getenv('ES_INDEX')
ES_PORT = os.getenv('ES_PORT', '9200')
ES_USER = os.getenv('ELASTIC_USERNAME')
ES_PASS = os.getenv('ELASTIC_PASSWORD')