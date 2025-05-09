from flask import request, jsonify
from flasgger import swag_from
import logging
from services.pdf_service import allowed_file, process_pdf
from services.elasticsearch_service import get_embedding_dimension
import tempfile
import os

logger = logging.getLogger("chat-ebook-ai")

def register_routes(app, es, embedder, model, ES_INDEX):
    @app.route('/upload', methods=['POST'])
    @swag_from({
        'tags': ['RAG Pipeline'],
        'summary': 'Upload a PDF file for processing',
        'description': 'Uploads a PDF, extracts text, generates embeddings, and indexes in Elasticsearch.',
        'consumes': ['multipart/form-data'],
        'parameters': [{'name': 'file', 'in': 'formData', 'type': 'file', 'required': True}],
        'responses': {
            '200': {'description': 'Indexed successfully', 'schema': {'properties': {'message': {'type': 'string'}, 'chunks': {'type': 'integer'}}}},
            '400': {'description': 'Invalid request', 'schema': {'properties': {'error': {'type': 'string'}}}},
            '500': {'description': 'Processing error', 'schema': {'properties': {'error': {'type': 'string'}}}}
        }
    })
    def upload_pdf():
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if not file or file.filename == '' or not allowed_file(file.filename):
            return jsonify({"error": "Invalid or missing file"}), 400
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                file.save(tmp.name)
                num_chunks = process_pdf(tmp.name, es, embedder, ES_INDEX)
            return jsonify({"message": "File indexed successfully", "chunks": num_chunks}), 200
        except Exception as e:
            logger.exception("Upload failed")
            return jsonify({"error": str(e)}), 500
        finally:
            if 'tmp' in locals() and os.path.exists(tmp.name): os.unlink(tmp.name)

    @app.route('/query', methods=['POST'])
    @swag_from({
        'tags': ['RAG Pipeline'],
        'summary': 'Query the processed PDF',
        'description': 'Submits a question and retrieves an answer using Gemini.',
        'consumes': ['application/json'],
        'parameters': [{'name': 'body', 'in': 'body', 'required': True, 'schema': {'properties': {'query': {'type': 'string'}}, 'required': ['query']}}],
        'responses': {
            '200': {'description': 'Successful response', 'schema': {'properties': {'answer': {'type': 'string'}, 'matched_chunks': {'type': 'array'}, 'scores': {'type': 'array'}, 'ids': {'type': 'array'}}}},
            '400': {'description': 'Invalid request', 'schema': {'properties': {'error': {'type': 'string'}}}},
            '500': {'description': 'Error processing query', 'schema': {'properties': {'error': {'type': 'string'}}}}
        }
    })
    def query():
        data = request.get_json() or {}
        q = data.get('query')
        if not q:
            return jsonify({"error": "Query is required"}), 400
        try:
            q_vec = embedder.encode([q], convert_to_tensor=False, normalize_embeddings=True)[0].tolist()
            k = 3
            body = {
                "size": k,
                "query": {
                    "knn": {
                        "field": "embedding",
                        "query_vector": q_vec,
                        "k": k,
                        "num_candidates": 100
                    }
                }
            }
            res = es.search(index=ES_INDEX, body=body)
            hits = res['hits']['hits']
            matched = [h['_source']['chunk'] for h in hits]
            scores = [h['_score'] for h in hits]
            ids = [int(h['_id']) for h in hits]
            context = "\n\n".join([f"Chunk {i+1}: {c}" for i, c in enumerate(matched)])
            prompt = (
                "You are an expert assistant answering questions based on provided document excerpts and your own knowledge. "
                "First, review the context and, if it’s relevant, use it to inform your answer. "
                "If the context is insufficient or unrelated to the question, acknowledge this and then answer fully using your general expertise. "
                "Always provide a clear, accurate, and concise response.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {q}\n\n"
                "Answer:"
            )
            logger.info("Prompt: %s", prompt)
            response = model.generate_content(prompt)
            answer = response.text or "No answer generated."
            return jsonify({"answer": answer, "matched_chunks": matched, "scores": scores, "ids": ids}), 200
        except Exception as e:
            logger.exception("Query failed")
            return jsonify({"error": str(e)}), 500

    @app.route('/inspect', methods=['GET'])
    @swag_from({
        'tags': ['RAG Pipeline'],
        'summary': 'Inspect index',
        'description': 'Returns count and sample entries.',
        'responses': {
            '200': {'description': 'Index data', 'schema': {'properties': {'ntotal': {'type': 'integer'}, 'dimension': {'type': 'integer'}, 'entries': {'type': 'array'}}}},
            '400': {'description': 'No index', 'schema': {'properties': {'error': {'type': 'string'}}}},
            '500': {'description': 'Error inspecting index', 'schema': {'properties': {'error': {'type': 'string'}}}}
        }
    })
    def inspect_index():
        try:
            if not es.indices.exists(index=ES_INDEX):
                return jsonify({"error": "Index does not exist"}), 400
            total = es.count(index=ES_INDEX)['count']
            resp = es.search(index=ES_INDEX, size=30)
            dimension = get_embedding_dimension(es, ES_INDEX)
            if dimension is None:
                return jsonify({"error": "Embedding dimension not found"}), 500

            entries = []
            for h in resp['hits']['hits']:
                src = h.get('_source', {})
                entry = {
                    'id': int(h['_id']),
                    'chunk': src.get('chunk', '[No chunk]'),
                    'embedding_preview': src.get('embedding', [])[:5]
                }
                entries.append(entry)

            return jsonify({'ntotal': total, 'dimension': dimension, 'entries': entries}), 200
        except Exception as e:
            logger.exception("Inspect failed")
            return jsonify({"error": str(e)}), 500