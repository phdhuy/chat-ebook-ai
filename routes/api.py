from flask import request, jsonify
from flasgger import swag_from
import logging
from services.pdf_service import allowed_file, process_pdf
from services.elasticsearch_service import get_embedding_dimension
import tempfile
import os
import re
import uuid

logger = logging.getLogger("chat-ebook-ai")

def register_routes(app, es, embedder, model, ES_INDEX):
    @app.route('/upload', methods=['POST'])
    @swag_from({
        'tags': ['RAG Pipeline'],
        'summary': 'Upload a PDF file for processing',
        'description': 'Uploads a PDF, extracts text, generates embeddings, and indexes in Elasticsearch with a conversation ID.',
        'consumes': ['multipart/form-data'],
        'parameters': [
            {'name': 'file', 'in': 'formData', 'type': 'file', 'required': True},
            {'name': 'conversation_id', 'in': 'formData', 'type': 'string', 'required': True,
             'description': 'UUID of the conversation'}
        ],
        'responses': {
            '200': {'description': 'Indexed successfully', 'schema': {
                'properties': {'message': {'type': 'string'}, 'chunks': {'type': 'integer'},
                               'conversation_id': {'type': 'string'}}}},
            '400': {'description': 'Invalid request', 'schema': {'properties': {'error': {'type': 'string'}}}},
            '500': {'description': 'Processing error', 'schema': {'properties': {'error': {'type': 'string'}}}}
        }
    })
    def upload_pdf():
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        conversation_id = request.form.get('conversation_id')

        if not file or file.filename == '' or not allowed_file(file.filename):
            return jsonify({"error": "Invalid or missing file"}), 400

        try:
            uuid_obj = uuid.UUID(conversation_id, version=4)
            conversation_id = str(uuid_obj)  # Ensure it's in string format
        except ValueError:
            return jsonify({"error": "Conversation ID must be a valid UUID"}), 400

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                file.save(tmp.name)
                num_chunks = process_pdf(tmp.name, es, embedder, ES_INDEX, conversation_id)
            return jsonify({
                "message": "File indexed successfully",
                "chunks": num_chunks,
                "conversation_id": conversation_id
            }), 200
        except Exception as e:
            logger.exception("Upload failed")
            return jsonify({"error": str(e)}), 500
        finally:
            if 'tmp' in locals() and os.path.exists(tmp.name):
                os.unlink(tmp.name)

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
            k = 5
            body = {
                "size": k,
                "query": {
                    "bool": {
                        "should": [
                            {
                                "match": {
                                    "chunk": {
                                        "query": q,
                                        "boost": 0.5
                                    }
                                }
                            },
                            {
                                "knn": {
                                    "field": "embedding",
                                    "query_vector": q_vec,
                                    "k": k,
                                    "num_candidates": 200,
                                    "boost": 1.0
                                }
                            }
                        ]
                    }
                }
            }

            res = es.search(index=ES_INDEX, body=body)
            hits = res['hits']['hits']

            matched = []
            for i, h in enumerate(hits):
                chunk = h['_source']['chunk']
                page = h['_source']['page']
                spans = h['_source'].get('spans', [])
                matched.append({
                    "id": i + 1,
                    "text": chunk,
                    "page": page,
                    "spans": spans
                })

            context = "\n".join([f"- [{m['id']}] (Page {m['page']}): {m['text']}" for m in matched])

            prompt = (
                "You are an expert assistant tasked with answering questions accurately and concisely using provided document excerpts when relevant. "
                "Follow these instructions:\n"
                "1. Use information from the document excerpts if they directly address the question, citing them as [number] where number is the excerpt label (e.g., [1]).\n"
                "2. If the excerpts are not relevant or insufficient, rely solely on your knowledge to provide an accurate answer.\n"
                "3. Reason through the question logically, considering the excerpts and your expertise, to ensure correctness.\n"
                "4. Output only the answer to the question, without including this prompt, instructions, or any extraneous information.\n"
                "5. If no answer can be derived, state 'Insufficient information to answer the question'.\n\n"
                f"Document Excerpts:\n{context}\n\n"
                f"Question: {q}"
            )

            response = model.generate_content(prompt)
            answer = response.text or "No answer generated."

            cited_ids = re.findall(r'\[(\d+)\]', answer)
            cited_excerpts = [m for m in matched if str(m['id']) in cited_ids]

            return jsonify({
                "answer": answer,
                "cited_excerpts": [
                    {
                        "id": e['id'],
                        "text": e['text'],
                        "page": e['page'],
                        "spans": e['spans']
                    } for e in cited_excerpts
                ]
            }), 200

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

    @app.route('/summarize', methods=['POST'])
    @swag_from({
        'tags': ['RAG Pipeline'],
        'summary': 'Summarize an eBook based on conversation ID',
        'description': 'Summarizes the content of an eBook associated with the given conversation ID.',
        'parameters': [
            {'name': 'body', 'in': 'body', 'required': True, 'schema': {
                'type': 'object',
                'properties': {
                    'conversation_id': {'type': 'string', 'description': 'UUID of the conversation'}
                }
            }}
        ],
        'responses': {
            '200': {'description': 'Summary generated', 'schema': {'properties': {'summary': {'type': 'string'},
                                                                                  'cited_pages': {'type': 'array',
                                                                                                  'items': {
                                                                                                      'type': 'integer'}}}}},
            '400': {'description': 'Invalid request', 'schema': {'properties': {'error': {'type': 'string'}}}},
            '404': {'description': 'No content found', 'schema': {'properties': {'error': {'type': 'string'}}}},
            '500': {'description': 'Summarization error', 'schema': {'properties': {'error': {'type': 'string'}}}}
        }
    })
    def summarize_ebook():
        data = request.get_json() or {}
        conversation_id = data.get('conversation_id')
        logger.info("conversation_id: %s", conversation_id)

        try:
            uuid_obj = uuid.UUID(conversation_id, version=4)
            conversation_id = str(uuid_obj)
        except ValueError:
            return jsonify({"error": "Conversation ID must be a valid UUID"}), 400

        try:
            body = {
                "size": 50,
                "query": {
                    "term": {"conversation_id.keyword": conversation_id}
                },
                "sort": [{"page": {"order": "asc"}}]
            }
            res = es.search(index=ES_INDEX, body=body)
            hits = res['hits']['hits']
            if not hits:
                return jsonify({"error": "No chunks found for this conversation ID"}), 404

            chunks = []
            for hit in hits:
                chunk = hit['_source']['chunk']
                page = hit['_source']['page']
                chunks.append({"text": chunk, "page": page})

            context = "\n".join([f"- (Page {c['page']}): {c['text']}" for c in chunks])

            system_message = (
                "You are an expert summarization assistant. "
                "When asked to summarize, only ever output the requested summary — "
                "do not include the prompt or any instructions in your response."
            )

            user_message = (
                "Please summarize the following eBook excerpt in 4–8 sentences, "
                "including page citations like [Page X] for any facts you pull directly. "
                "Focus on the main ideas and avoid excessive detail.\n\n"
                f"eBook Content:\n{context}\n\n"
                "====\nSummary:"
            )

            prompt = system_message + "\n\n" + user_message

            response = model.generate_content(prompt)
            summary = response.text or "No summary generated."

            cited_pages = set()
            page_pattern = r'\[Page (\d+)\]'
            for match in re.finditer(page_pattern, summary):
                cited_pages.add(int(match.group(1)))

            return jsonify({
                "summary": summary,
                "cited_pages": sorted(list(cited_pages))
            }), 200

        except Exception as e:
            logger.exception("Summarization failed")
            return jsonify({"error": str(e)}), 500