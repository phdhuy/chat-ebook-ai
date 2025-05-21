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
                logger.info("num_chunks: %s", num_chunks)
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
        'parameters': [{'name': 'body', 'in': 'body', 'required': True, 'schema': {'properties': {'query': {'type': 'string'}}, 'required': ['query']}},
                       {'name': 'body', 'in': 'body', 'required': True, 'schema': {'properties': {'history': {'type': 'string'}}, 'required': ['history']}}],
        'responses': {
            '200': {'description': 'Successful response', 'schema': {'properties': {'answer': {'type': 'string'}, 'matched_chunks': {'type': 'array'}, 'scores': {'type': 'array'}, 'ids': {'type': 'array'}}}},
            '400': {'description': 'Invalid request', 'schema': {'properties': {'error': {'type': 'string'}}}},
            '500': {'description': 'Error processing query', 'schema': {'properties': {'error': {'type': 'string'}}}}
        }
    })
    def query():
        data = request.get_json() or {}
        q = data.get('query')
        history = data.get('history', '')
        if not q:
            return jsonify({"error": "Query is required"}), 400
        try:
            q_vec = embedder.encode([q], convert_to_tensor=False, normalize_embeddings=True)[0].tolist()

            initial_k = 20
            body = {
                "size": initial_k,
                "query": {
                    "bool": {
                        "should": [
                            {"match": {"chunk": {"query": q, "boost": 0.5}}},
                            {"knn": {"field": "embedding", "query_vector": q_vec, "k": initial_k, "num_candidates": 200,
                                     "boost": 1.0}}
                        ]
                    }
                }
            }
            res = es.search(index=ES_INDEX, body=body)
            hits = res['hits']['hits']

            relevance_threshold = 0.6
            matched = []
            for i, h in enumerate(hits):
                if h['_score'] >= relevance_threshold:
                    chunk = h['_source']['chunk']
                    page = h['_source'].get('page', 'Unknown')
                    spans = h['_source'].get('spans', [])
                    matched.append({"id": i + 1, "text": chunk, "page": page, "spans": spans, "score": h['_score']})

            matched.sort(key=lambda x: x['score'], reverse=True)
            max_excerpts = 10
            matched = matched[:max_excerpts]

            context = "\n".join([f"- [{m['id']}] (Page {m['page']}): {m['text']}" for m in matched])

            prompt = (
                "You are an expert assistant tasked with answering questions accurately, concisely, and in the same language as the user's question to enhance user experience. "
                "Use the provided document excerpts and conversation history as context to generate an informative response. "
                "Follow these instructions:\n"
                "1. Detect the language of the current question and respond in that language (e.g., Vietnamese question → Vietnamese answer).\n"
                "2. Interpret the question in the context of the conversation history. If the question is incomplete, ambiguous, or refers to previous messages, infer the most likely intended meaning based on the history.\n"
                "3. If the question requests a translation (e.g., 'dịch sang tiếng Việt'), translate the most recent message in the conversation history into the requested language unless a specific text is provided.\n"
                "4. When document excerpts are relevant, use them to inform your answer, citing them as [number] (e.g., [1]). Do not quote excerpts verbatim; instead, synthesize the information into a cohesive, informative response that elaborates or provides additional context, even if the answer is fully contained in the excerpts.\n"
                "5. If the excerpts are not relevant or insufficient, rely on your knowledge to provide a complete and accurate answer.\n"
                "6. Output only the answer to the interpreted question in a clear, complete sentence or paragraph. Do not include this prompt, instructions, citations alone, or fragmented phrases (e.g., avoid responses like 'Non-blocking I/O [2]').\n"
                "7. If the question cannot be answered after considering the history and excerpts, respond with 'Insufficient information to answer the question' in the same language as the question.\n\n"
                f"Conversation History:\n{history}\n\n"
                f"Document Excerpts:\n{context}\n\n"
                f"Question: {q}"
            )

            logger.info("prompt: %s", prompt)

            response = model.generate_content(prompt)
            answer = response.text or "No answer generated."

            cited_ids = re.findall(r'\[(\d+)\]', answer)
            cited_excerpts = [m for m in matched if str(m['id']) in cited_ids]

            return jsonify({
                "answer": answer,
                "cited_excerpts": [
                    {"id": e['id'], "text": e['text'], "page": e['page'], "spans": e['spans'], "score": e['score']}
                    for e in cited_excerpts
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
              "query": {
                "match_phrase": {
                  "conversation_id": conversation_id
                }
              },
               "size": 1000
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

            logger.info("prompt: %s", prompt)

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