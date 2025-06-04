from flask import request, jsonify
from flasgger import swag_from
import logging
from services.pdf_service import allowed_file, process_pdf
from services.elasticsearch_service import get_embedding_dimension, retrieve_full_text_from_es
import tempfile
import os
import re
import uuid
import fitz
import json

logger = logging.getLogger("chat-ebook-ai")

def register_routes(app, es, embedder, model, ES_INDEX):
    @app.route('/upload', methods=['POST'])
    @swag_from({
        'tags': ['RAG Pipeline'],
        'summary': 'Upload a PDF file for processing',
        'description': 'Uploads a PDF file, extracts its text, generates embeddings, and indexes the content in Elasticsearch using a conversation ID.',
        'consumes': ['multipart/form-data'],
        'parameters': [
            {
                'name': 'file',
                'in': 'formData',
                'type': 'file',
                'required': True,
                'description': 'The PDF file to upload for processing.'
            },
            {
                'name': 'conversation_id',
                'in': 'formData',
                'type': 'string',
                'required': True,
                'description': 'UUID of the conversation to associate with the indexed content.',
                'format': 'uuid'
            }
        ],
        'responses': {
            '200': {
                'description': 'File indexed successfully',
                'schema': {
                    'type': 'object',
                    'properties': {
                        'message': {'type': 'string', 'example': 'File indexed successfully'},
                        'chunks': {'type': 'integer', 'example': 50},
                        'conversation_id': {'type': 'string', 'format': 'uuid',
                                            'example': '123e4567-e89b-12d3-a456-426614174000'}
                    }
                }
            },
            '400': {
                'description': 'Invalid request (e.g., missing file or invalid conversation ID)',
                'schema': {
                    'type': 'object',
                    'properties': {
                        'error': {'type': 'string', 'example': 'Invalid or missing file'}
                    }
                }
            },
            '500': {
                'description': 'Server error during processing',
                'schema': {
                    'type': 'object',
                    'properties': {
                        'error': {'type': 'string', 'example': 'Internal server error'}
                    }
                }
            }
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
            conversation_id = str(uuid_obj)
        except ValueError:
            return jsonify({"error": "Conversation ID must be a valid UUID"}), 400

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                file.save(tmp.name)
                num_chunks = process_pdf(tmp.name, es, embedder, ES_INDEX, conversation_id)
                logger.info("num_chunks: %s", num_chunks)

                es.indices.refresh(index=ES_INDEX)

                full_text = retrieve_full_text_from_es(conversation_id, es, ES_INDEX)
                if not full_text:
                    raise ValueError("No text retrieved from Elasticsearch")

                system_prompt = (
                    "Summarize the following text in about 3 sentences, focusing on the main themes. "
                    "Then, generate five question prompts for user engagement, formatted on separate lines starting with '1.' and '2.'.\n"
                    "Provide the summary first, followed by a blank line, then the questions:\n\n"
                    f"{full_text[:1000]}"
                )

                response = model.generate_content(system_prompt)

            return jsonify({
                "message": "File indexed successfully",
                "chunks": num_chunks,
                "conversation_id": conversation_id,
                "answer": response.text,
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
        'parameters': [
            {
                'name': 'body',
                'in': 'body',
                'required': True,
                'schema': {
                    'type': 'object',
                    'properties': {
                        'query': {'type': 'string'},
                        'history': {'type': 'string'},
                        'conversation_id': {'type': 'string'}
                    },
                    'required': ['query', 'conversation_id']
                }
            }
        ],
        'responses': {
            '200': {
                'description': 'Successful response',
                'schema': {
                    'type': 'object',
                    'properties': {
                        'answer': {'type': 'string'},
                        'matched_chunks': {'type': 'array', 'items': {'type': 'string'}},
                        'scores': {'type': 'array', 'items': {'type': 'number'}},
                        'ids': {'type': 'array', 'items': {'type': 'string'}}
                    }
                }
            },
            '400': {
                'description': 'Invalid request',
                'schema': {
                    'type': 'object',
                    'properties': {
                        'error': {'type': 'string'}
                    }
                }
            },
            '500': {
                'description': 'Error processing query',
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
        data = request.get_json() or {}
        q = data.get('query')
        history = data.get('history', '')
        conversation_id = data.get('conversation_id')

        if not q:
            return jsonify({"error": "Query is required"}), 400
        if not conversation_id:
            return jsonify({"error": "Conversation ID is required"}), 400

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

            relevance_threshold = 2.5
            matched = []
            for i, h in enumerate(hits):
                if h['_score'] >= relevance_threshold and h['_source']['conversation_id'] == conversation_id:
                    chunk = h['_source']['chunk']
                    page = h['_source'].get('page', 'Unknown')
                    spans = h['_source'].get('spans', [])
                    matched.append({"id": i + 1, "text": chunk, "page": page, "spans": spans, "score": h['_score']})

            matched.sort(key=lambda x: x['score'], reverse=True)
            max_excerpts = 5
            matched = matched[:max_excerpts]

            context = "\n".join([f"- [{m['id']}] (Page {m['page']}): {m['text']}" for m in matched])

            prompt = (
                "You are an expert assistant responsible for providing accurate and concise answers in the same language as the user's question to optimize user experience. "
                "Utilize the provided document excerpts and conversation history as context to craft an informative response, adhering to the following guidelines:\n"
                "1. Identify the language of the current question and respond accordingly.\n"
                "2. Analyze the question in the context of the conversation history; if it is unclear, incomplete, or references prior messages, deduce the most likely intent based on the history.\n"
                "3. Incorporate relevant document excerpts into your answer by synthesizing the information into a cohesive response with additional context, citing them as [number] (e.g., [1]) only when the excerpts directly contribute to the answer.\n"
                "4. If document excerpts are irrelevant or insufficient, rely solely on your knowledge to deliver a complete and accurate response without mentioning or citing the excerpts.\n"
                "5. Provide only the answer to the interpreted question in a clear, complete sentence or paragraph, avoiding this prompt, standalone citations, or fragmented phrases (e.g., avoid 'Non-blocking I/O [2]').\n"
                "6. If the question cannot be answered based on the history and excerpts, respond with 'Insufficient information to answer the question' in the same language as the question.\n"
                "7. Do not include any citation markers if the excerpts are not used in the response to avoid user confusion\n\n"
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
                    {"id": e['id'], "text": e['text'], "page": e['page'], "score": e['score']}
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

    @app.route('/mindmap', methods=['POST'])
    @swag_from({
        'tags': ['RAG Pipeline'],
        'summary': 'Upload a PDF file to generate a mind map',
        'description': 'Uploads a PDF file and generates a mind map based on its table of contents (outlines). The PDF must contain a table of contents (outlines) for the mind map to be generated. If no TOC is found, an error will be returned.',
        'consumes': ['multipart/form-data'],
        'parameters': [
            {
                'name': 'file',
                'in': 'formData',
                'type': 'file',
                'required': True,
                'description': 'The PDF file to upload for processing. The PDF must have a table of contents (outlines) set.'
            }
        ],
        'responses': {
            '200': {
                'description': 'Mind map generated successfully',
                'schema': {
                    'type': 'object',
                    'properties': {
                        'nodes': {'type': 'array', 'items': {'type': 'object'}},
                        'edges': {'type': 'array', 'items': {'type': 'object'}}
                    }
                }
            },
            '400': {
                'description': 'Invalid request (e.g., missing file, invalid file type, or no table of contents in the PDF)',
                'schema': {
                    'type': 'object',
                    'properties': {
                        'error': {'type': 'string',
                                  'example': 'The uploaded PDF does not contain a table of contents. Please ensure the PDF has outlines set.'}
                    }
                }
            },
            '500': {
                'description': 'Server error during processing',
                'schema': {
                    'type': 'object',
                    'properties': {
                        'error': {'type': 'string', 'example': 'Internal server error'}
                    }
                }
            }
        }
    })
    def generate_mindmap():
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload a PDF'}), 400

        filepath = '/tmp/uploaded.pdf'
        file.save(filepath)

        try:
            doc = fitz.open(filepath)
            toc = doc.get_toc()

            if not toc:
                text = ""
                for page in doc:
                    text += page.get_text("text") + "\n"

                prompt = f"""
                Generate a structured table of contents (TOC) from the following PDF text as a JSON array of arrays: [[level, "title", page]].

                - 'level': Integer (1 for main, 2 for sub, etc.).
                - "title": Concise section title.
                - page: Estimated integer page number (distribute across {doc.page_count} pages).

                Reflect the document's hierarchical structure. Return an empty array if the text is insufficient.

                Text:
                {text}

                Output:
                """

                response = model.generate_content(prompt)
                if not response:
                    return jsonify({'error': 'Failed to generate table of contents for the PDF'}), 400

                response_text = response.text if hasattr(response, 'text') else str(response)

                try:
                    json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group()
                        toc = json.loads(json_str)
                    else:
                        toc = json.loads(response_text.strip())

                    if not isinstance(toc, list):
                        raise ValueError("Response is not a list")

                    for item in toc:
                        if not isinstance(item, list) or len(item) != 3:
                            raise ValueError("Invalid TOC format")

                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Error parsing AI response: {e}")
                    print(f"Response text: {response_text}")
                    return jsonify({'error': 'Failed to parse AI-generated table of contents'}), 400

            title = doc.metadata.get('title', 'Document')

            tree = {'root': []}
            nodes_info = {'root': {'label': title, 'page': None}}
            parents = ['root']
            current_id = 1

            for entry in toc:
                try:
                    level, title_text, page = entry

                    if not isinstance(level, int):
                        level = int(level)

                    if not isinstance(page, int):
                        page = int(page)

                except (ValueError, TypeError) as e:
                    print(f"Error processing TOC entry {entry}: {e}")
                    continue

                while len(parents) > level:
                    parents.pop()

                while len(parents) < level:
                    dummy_id = f'dummy{current_id}'
                    current_id += 1
                    parent_id = parents[-1]
                    if parent_id not in tree:
                        tree[parent_id] = []
                    tree[parent_id].append(dummy_id)
                    nodes_info[dummy_id] = {'label': f'Section {len(parents)}', 'page': page}
                    parents.append(dummy_id)

                parent_id = parents[-1]
                node_id = f'node{current_id}'
                current_id += 1

                if parent_id not in tree:
                    tree[parent_id] = []
                tree[parent_id].append(node_id)
                nodes_info[node_id] = {'label': title_text, 'page': page}
                parents.append(node_id)

            nodes = []
            for node_id in nodes_info:
                label = nodes_info[node_id]['label']
                page = nodes_info[node_id]['page']

                if node_id == 'root':
                    node_type = 'input'
                elif node_id in tree and tree[node_id]:
                    node_type = None
                else:
                    node_type = 'output'

                node = {
                    'id': node_id,
                    'data': {'label': label, 'page': page},
                    'position': {'x': 0, 'y': 0}
                }
                if node_type:
                    node['type'] = node_type
                nodes.append(node)

            edges = []
            for parent_id in tree:
                for child_id in tree[parent_id]:
                    edge_id = f'e{parent_id}-{child_id}'
                    edges.append({
                        'id': edge_id,
                        'source': parent_id,
                        'target': child_id,
                        'type': 'smoothstep',
                        'animated': False
                    })

            return jsonify({'nodes': nodes, 'edges': edges})

        except Exception as e:
            print(f"Unexpected error: {e}")
            return jsonify({'error': str(e)}), 500

        finally:
            if 'doc' in locals():
                doc.close()
            if os.path.exists(filepath):
                os.remove(filepath)