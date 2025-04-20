from flask import Flask, request, jsonify
from flasgger import Swagger, swag_from
from flask_cors import CORS
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
import os
from dotenv import load_dotenv
import tempfile
from pdf2image import convert_from_path
import pytesseract
import logging
from PIL import Image
import nltk
nltk.download('punkt_tab')

app = Flask(__name__)
swagger = Swagger(app)

# Configure CORS
CORS(app)

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not set in environment variables")
genai.configure(api_key=GEMINI_API_KEY)

# Initialize models
embedder = SentenceTransformer('all-MiniLM-L6-v2')
model = genai.GenerativeModel('gemini-1.5-flash')

# Global variables for FAISS index and chunks
index = None
chunks = []

# Configure logging (once, ideally at the top of your app.py)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'


def process_pdf(filepath):
    global index, chunks

    # Try extracting text via PyPDF2
    with open(filepath, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted

    if not text.strip():
        # Fallback to OCR
        print("No text found with PyPDF2. Using OCR fallback...")
        images = convert_from_path(filepath)
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img)

    if not text.strip():
        raise ValueError("No extractable text found in PDF (even with OCR).")

    # Split text into sentences using NLTK
    sentences = nltk.sent_tokenize(text)

    # Combine sentences into chunks (you can control the chunk size here)
    chunk_size = 5  # Adjust this based on the desired size of each chunk
    chunks = [' '.join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]

    if not chunks:
        raise ValueError("Text found but no valid chunks.")

    embeddings = embedder.encode(chunks, convert_to_tensor=False)
    embeddings = np.array(embeddings, dtype='float32')
    if embeddings.ndim == 1:
        embeddings = embeddings[np.newaxis, :]
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)



@app.route('/upload', methods=['POST'])
@swag_from({
    'tags': ['RAG Pipeline'],
    'summary': 'Upload a PDF file for processing',
    'description': 'Uploads a PDF file, extracts text, generates embeddings, and stores them in FAISS.',
    'consumes': ['multipart/form-data'],
    'parameters': [
        {
            'name': 'file',
            'in': 'formData',
            'type': 'file',
            'required': True,
            'description': 'PDF file to upload'
        }
    ],
    'responses': {
        '200': {
            'description': 'File processed successfully',
            'schema': {
                'type': 'object',
                'properties': {
                    'message': {'type': 'string'}
                }
            }
        },
        '400': {
            'description': 'Invalid request or file type',
            'schema': {
                'type': 'object',
                'properties': {
                    'error': {'type': 'string'}
                }
            }
        }
    }
})
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type; only PDF allowed"}), 400

    try:
        # Use delete=False to avoid early deletion locks on some OSes
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            file.save(tmp.name)
            app.logger.debug(f"Saved upload to temp file: {tmp.name}")
        process_pdf(tmp.name)
        app.logger.info(f"Processed PDF, built FAISS index of size {index.ntotal}")
        return jsonify({"message": "File processed successfully", "chunks": len(chunks)}), 200

    except Exception as e:
        # Log full traceback to console
        app.logger.exception("Failed to process uploaded PDF")
        return jsonify({"error": f"Failed to process PDF: {str(e)}"}), 500


@app.route('/query', methods=['POST'])
@swag_from({
    'tags': ['RAG Pipeline'],
    'summary': 'Query the processed PDF',
    'description': 'Submits a question and receives an answer based on the uploaded PDF content using Gemini API.',
    'consumes': ['application/json'],
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'The question to ask about the PDF content'
                    }
                },
                'required': ['query']
            }
        }
    ],
    'responses': {
        '200': {
            'description': 'Successful response with the answer',
            'schema': {
                'type': 'object',
                'properties': {
                    'answer': {'type': 'string'}
                }
            }
        },
        '400': {
            'description': 'Invalid request or no PDF processed',
            'schema': {
                'type': 'object',
                'properties': {
                    'error': {'type': 'string'}
                }
            }
        },
        '500': {
            'description': 'Error communicating with Gemini API',
            'schema': {
                'type': 'object',
                'properties': {
                    'error': {'type': 'string'}
                }
            }
        }
    }
})
def query():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Query is required"}), 400

    query_text = data['query']
    if index is None:
        return jsonify({"error": "No PDF processed yet"}), 400

    try:
        # Create embedding for the query
        query_embedding = embedder.encode([query_text], convert_to_tensor=False)

        # Search the FAISS index
        k = 3  # Adjust the number of results to retrieve
        distances, indices = index.search(np.array(query_embedding, dtype='float32'), k)

        # Log and return results
        # Ensure that the chunks and distances are sorted by the distances
        sorted_indices = sorted(zip(distances[0], indices[0]), key=lambda x: x[0])

        # Extract the sorted chunks and distances
        sorted_chunks = [chunks[idx] for _, idx in sorted_indices]

        # Convert distances and indices to Python native types (float and int)
        sorted_distances = [float(dist) for dist, _ in sorted_indices]  # Convert to float
        sorted_indices = [int(idx) for _, idx in sorted_indices]  # Convert to int

        return jsonify({
            "matched_chunks": sorted_chunks,
            "distances": sorted_distances,
            "indices": sorted_indices
        }), 200

    except Exception as e:
        return jsonify({"error": f"Error processing query: {str(e)}"}), 500


@app.route('/inspect', methods=['GET'])
@swag_from({
    'tags': ['RAG Pipeline'],
    'summary': 'Inspect FAISS index contents',
    'description': 'Returns metadata and a preview of stored vectors and their text chunks.',
    'responses': {
        '200': {
            'description': 'Index inspection data',
            'schema': {
                'type': 'object',
                'properties': {
                    'ntotal': {'type': 'integer'},
                    'dimension': {'type': 'integer'},
                    'entries': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'id': {'type': 'integer'},
                                'chunk': {'type': 'string'},
                                'embedding_preview': {
                                    'type': 'array',
                                    'items': {'type': 'number'}
                                }
                            }
                        }
                    }
                }
            }
        },
        '400': {
            'description': 'No index available',
            'schema': {'type': 'object', 'properties': {'error': {'type': 'string'}}}
        }
    }
})
def inspect_index():
    if index is None:
        return jsonify({'error': 'No FAISS index built yet'}), 400

    # Basic metadata
    try:
        dim = index.d
    except AttributeError:
        dim = index.dim()
    total = int(index.ntotal)

    # Build a complete preview of all entries
    entries = []
    for i in range(total):
        vec = index.reconstruct(i)  # requires a Flat index
        entries.append({
            'id': i,
            'chunk': chunks[i],  # Assuming chunks[] contains the corresponding text
            'embedding_preview': vec.tolist()[:5]  # Limit preview to first 5 elements of the embedding
        })

    return jsonify({
        'ntotal': total,
        'dimension': dim,
        'entries': entries
    }), 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)