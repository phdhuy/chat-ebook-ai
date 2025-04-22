from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from config import GEMINI_API_KEY

def initialize_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    return embedder, model