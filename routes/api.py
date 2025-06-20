from flask import request, jsonify
from flasgger import swag_from
import logging
from services.pdf_service import allowed_file, process_pdf
from services.elasticsearch_service import get_embedding_dimension
import tempfile
import os
import re
import uuid
import fitz
import json
from ragas.metrics import faithfulness, answer_similarity, context_precision, answer_relevancy, answer_correctness
from ragas import evaluate
from datasets import Dataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

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
                chunks = process_pdf(tmp.name, es, embedder, ES_INDEX, conversation_id, return_chunks=True)
                full_text = " ".join([chunk["text"] for chunk in chunks])
                if not full_text:
                    raise ValueError("No text available for summarization")

                system_prompt = (
                    "Summarize the following text in about 3 sentences, focusing on the main themes. "
                    f"{full_text[:1000]}"
                )

                response = model.generate_content(system_prompt)

            return jsonify({
                "message": "File indexed successfully",
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
    def query_rag(query="", history="", conversation_id=""):
        # data = request.get_json() or {}
        # q = data.get('query')
        # history = data.get('history', '')
        # conversation_id = data.get('conversation_id')

        q = query

        if not q:
            return jsonify({"error": "Query is required"}), 400
        try:
            hyde_prompt = (
                f"Create a concise, factual, and detailed hypothetical document answering the question: '{q}'. "
                f"Use precise terminology and relevant examples typical of authoritative sources. "
                f"Keep the response under 500 tokens to ensure brevity and focus."
            )
            hyde_response = model.generate_content(hyde_prompt)
            logger.info("HyDE response: %s", hyde_response)
            hypothetical_doc = hyde_response.text.strip()
            q_vec = embedder.encode([hypothetical_doc], convert_to_tensor=False, normalize_embeddings=True)[0].tolist()

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
                if conversation_id:
                    if h['_score'] >= relevance_threshold and h['_source']['conversation_id'] == conversation_id:
                        chunk = h['_source']['chunk']
                        page = h['_source'].get('page', 'Unknown')
                        spans = h['_source'].get('spans', [])
                        matched.append({"id": i + 1, "text": chunk, "page": page, "spans": spans, "score": h['_score']})
                else:
                    if h['_score'] >= relevance_threshold:
                        chunk = h['_source']['chunk']
                        page = h['_source'].get('page', 'Unknown')
                        spans = h['_source'].get('spans', [])
                        matched.append({"id": i + 1, "text": chunk, "page": page, "spans": spans, "score": h['_score']})

            matched.sort(key=lambda x: x['score'], reverse=True)
            max_excerpts = 15
            matched = matched[:max_excerpts]

            context = "\n".join([f"- [Page {m['page']}]: {m['text']}" for m in matched])

            prompt = (
                "You are an expert assistant responsible for providing accurate and concise answers in the same language as the user's question to optimize user experience. "
                "Utilize the provided document excerpts and conversation history as context to craft an informative response, adhering to the following guidelines:\n"
                "1. Identify the language of the current question and respond accordingly.\n"
                "2. Analyze the question in the context of the conversation history; if it is unclear, incomplete, or references prior messages, deduce the most likely intent based on the history.\n"
                "3. Incorporate relevant document excerpts into your answer by synthesizing the information into a cohesive response with additional context, citing them as [page number] (e.g., [58]) only when the excerpts directly contribute to the answer. Each document excerpt is labeled as `- [Page page]: text`, so use the page number for citations.\n"
                "4. If document excerpts are irrelevant or insufficient, rely solely on your knowledge to deliver a complete and accurate response without mentioning or citing the excerpts.\n"
                "5. Provide only the answer to the interpreted question in a clear, complete sentence or paragraph, avoiding this prompt, standalone citations, or fragmented phrases (e.g., avoid 'IAM [58]').\n"
                "6. If the question cannot be answered based on the history and excerpts, respond with 'Insufficient information to answer the question' in the same language as the question.\n"
                f"Conversation History:\n{history}\n\n"
                f"Document Excerpts:\n{context}\n\n"
                f"Question: {q}"
            )

            logger.info("prompt: %s", prompt)

            response = model.generate_content(prompt)
            answer = response.text or "No answer generated."

            cited_pages = re.findall(r'\[(\d+)\]', answer)
            cited_excerpts = [m for m in matched if str(m['page']) in cited_pages]

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
                "including page citations like [X] for any facts you pull directly. "
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

    @app.route('/evaluate', methods=['POST'])
    @swag_from({
        'tags': ['RAG Pipeline'],
        'summary': 'Evaluate RAG retrieval and generation performance',
        'description': (
                'Runs a suite of predefined queries through the RAG pipeline, compares the generated answers '
                'against ground-truth, and computes evaluation metrics: faithfulness, answer similarity, and context precision.'
        ),
        'consumes': ['application/json'],
        'responses': {
            '200': {
                'description': 'Evaluation completed successfully',
                'schema': {
                    'type': 'object',
                    'properties': {
                        'faithfulness': {'type': 'number', 'format': 'float'},
                        'answer_similarity': {'type': 'number', 'format': 'float'},
                        'context_precision': {'type': 'number', 'format': 'float'}
                    }
                }
            },
            '500': {
                'description': 'Internal server error during evaluation',
                'schema': {
                    'type': 'object',
                    'properties': {
                        'error': {'type': 'string', 'example': 'Internal server error'}
                    }
                }
            }
        }
    })
    def evaluate_rag():
        google_api_key = os.getenv("GEMINI_API_KEY")
        if not google_api_key:
            logger.error("GEMINI_API_KEY environment variable not set")
            return jsonify({
                "error": "GEMINI_API_KEY environment variable not set. Get your key from https://makersuite.google.com/app/apikey"}), 500

        try:
            gemini_llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=google_api_key,
                temperature=0.1,
                max_output_tokens=2048,
                top_k=40,
                top_p=0.95
            )

            # Setup Gemini Embeddings
            gemini_embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",  # Use a more stable embedding model
                google_api_key=google_api_key
            )

            ragas_llm = LangchainLLMWrapper(gemini_llm)
            ragas_embeddings = LangchainEmbeddingsWrapper(gemini_embeddings)

            logger.info("Gemini LLM and embeddings initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {str(e)}")
            return jsonify({"error": f"Failed to initialize Gemini: {str(e)}"}), 500

        # Expanded evaluation questions and ground truth answers for better evaluation
        eval_questions = [
            # "What is the main disadvantage of using a separator-based storage format, such as CSV, in a database engine?",
            # "How does a fixed-size row storage format improve performance compared to a separator-based format?",
            # "What is a significant drawback of using fixed-size rows in a real database system?",
            # "What does TLV stand for, and what are the three components it represents?",
            "In the context of TLV encoding, how is an integer value like 3894 represented, assuming it is stored as an 8-byte integer?",
            "Why is a byte-by-byte scanning algorithm inefficient for large tables, and how does a row-by-row approach address this?"
        ]
        eval_answers = [
            # "The main disadvantage of a separator-based storage format, like CSV, is its slow performance due to linear time complexity, O(n). The engine must iterate through the data byte-by-byte because it does not know the length of rows or individual columns, leading to potentially millions of iterations for large datasets (e.g., 500MB table requiring up to 500,000,000 iterations).",
            # "A fixed-size row storage format improves performance by allowing the database engine to scan the table row-by-row instead of byte-by-byte. This reduces the number of iterations significantly (e.g., a 640MB table with 4 million rows requires 4 million iterations instead of 640 million), offering up to a 160x performance improvement.",
            # "A significant drawback of fixed-size rows is the large space requirement for text-based columns. In real database systems, columns like varchar (255 bytes), text (65,535 bytes), mediumtext (16,777,215 bytes), or longtext (4,294,967,295 bytes) can lead to substantial storage overhead, as each row must reserve the maximum possible size for these columns.",
            # "TLV stands for Type-Length-Value. The three components are: Type: Indicates the data type (e.g., 1 for integer, 2 for string). Length: Specifies the size of the value in bytes (e.g., 8 bytes for an integer). Value: The actual data being stored (e.g., 3894 for an integer or hello for a string).",
            "In TLV encoding, the integer value 3894 is represented as: Type: 1 (indicating an integer). Length: 8 (indicating 8 bytes). Value: 3894 (stored in little-endian binary format, occupying 8 bytes).",
            "A byte-by-byte scanning algorithm is inefficient because it requires examining every byte of a table, leading to O(n) complexity and potentially billions of iterations for large tables (e.g., 640MB). A row-by-row approach, enabled by fixed-size or TLV formats, reduces iterations by processing entire rows at once, significantly lowering the computational cost (e.g., 4 million iterations for 4 million rows)."
        ]

        # Log all evaluation questions and expected answers
        logger.info("=== EVALUATION QUESTIONS AND EXPECTED ANSWERS ===")
        for i, (question, expected_answer) in enumerate(zip(eval_questions, eval_answers), 1):
            logger.info(f"Q{i}: {question}")
            logger.info(f"Expected A{i}: {expected_answer}")
            logger.info("-" * 80)

        eval_data = []
        successful_queries = 0
        failed_queries = 0

        # Process each evaluation query
        for i, (query, ground_truth) in enumerate(zip(eval_questions, eval_answers), 1):
            try:
                logger.info(f"Processing evaluation query {i}/{len(eval_questions)}: {query}")

                # Query the RAG system
                response, status_code = query_rag(query)

                if status_code != 200:
                    error_msg = "Unknown error"
                    try:
                        error_msg = response.get_json().get('error', 'Unknown error')
                    except:
                        pass
                    logger.error(f"Query {i} '{query}' failed with status {status_code}: {error_msg}")
                    failed_queries += 1
                    continue

                # Extract response data
                data = response.get_json()
                answer = data.get("answer", "No answer generated")
                cited_excerpts = data.get("cited_excerpts", [])

                # Log the actual RAG response for this question
                logger.info(f"RAG Response for Q{i}:")
                logger.info(f"Answer: {answer}")
                logger.info(f"Number of cited excerpts: {len(cited_excerpts)}")

                # Process contexts from cited excerpts
                if not isinstance(cited_excerpts, list):
                    logger.warning(f"cited_excerpts for query {i} '{query}' is not a list: {cited_excerpts}")
                    contexts = []
                else:
                    contexts = [excerpt["text"] for excerpt in cited_excerpts if
                                isinstance(excerpt, dict) and "text" in excerpt]

                # Log contexts
                logger.info(f"Contexts for Q{i}: {len(contexts)} context(s) found")
                for j, context in enumerate(contexts, 1):
                    logger.info(f"Context {j}: {context[:200]}...")

                # Ensure we have meaningful contexts for evaluation
                if not contexts:
                    logger.warning(f"No valid contexts found for query {i} '{query}', skipping this query")
                    failed_queries += 1
                    continue

                # Validate that answer and contexts are meaningful
                if len(answer.strip()) < 10:
                    logger.warning(f"Answer too short for query {i} '{query}', skipping")
                    failed_queries += 1
                    continue

                # Create evaluation entry
                eval_entry = {
                    "question": query,
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": ground_truth
                }

                eval_data.append(eval_entry)
                successful_queries += 1
                logger.info(f"Successfully processed query {i}: {query}")
                logger.info("=" * 80)

            except Exception as e:
                logger.error(f"Error processing query {i} '{query}': {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                failed_queries += 1
                continue

        # Check if we have any evaluation data
        logger.info(f"Successfully processed {len(eval_data)} evaluation queries")
        if not eval_data:
            logger.error("No evaluation data collected - all queries failed")
            return jsonify({"error": "No evaluation data collected - all queries failed"}), 400

        if len(eval_data) < 2:
            logger.warning(f"Only {len(eval_data)} evaluation samples collected. Results may not be reliable.")

        logger.info(f"Collected {len(eval_data)} evaluation entries")

        try:
            # Convert list to Dataset object
            logger.info("Converting evaluation data to Dataset...")
            dataset = Dataset.from_list(eval_data)

            # Validate dataset
            required_columns = ["question", "answer", "contexts", "ground_truth"]
            missing_columns = [col for col in required_columns if col not in dataset.column_names]
            if missing_columns:
                logger.error(f"Missing required columns in dataset: {missing_columns}")
                return jsonify({"error": f"Missing required columns: {missing_columns}"}), 400

            logger.info(f"Dataset created successfully with columns: {dataset.column_names}")
            logger.info(f"Dataset size: {len(dataset)} samples")

            # Log sample data for debugging
            if len(dataset) > 0:
                sample = dataset[0]
                logger.info(f"Sample data - Question: {sample['question'][:100]}...")
                logger.info(f"Sample data - Answer: {sample['answer'][:100]}...")
                logger.info(f"Sample data - Contexts count: {len(sample['contexts'])}")
                logger.info(f"Sample data - Ground truth: {sample['ground_truth'][:100]}...")

            # Perform RAGAS evaluation with Gemini
            logger.info("Starting RAGAS evaluation with Gemini...")
            logger.info("This may take a few minutes depending on the data size...")

            # Use a more robust set of metrics
            evaluation_metrics = [faithfulness, answer_relevancy, context_precision, answer_correctness]

            results = evaluate(
                dataset=dataset,
                metrics=evaluation_metrics,
                llm=ragas_llm,
                embeddings=ragas_embeddings,
            )

            logger.info("RAGAS evaluation completed successfully")

            # Enhanced logging of raw results with question-answer pairs
            logger.info("=== RAW EVALUATION RESULTS ===")
            logger.info(f"Raw evaluation results object: {results}")
            logger.info(f"Results type: {type(results)}")

            # Log individual results for each question if available
            if hasattr(results, 'to_pandas'):
                df = results.to_pandas()
                logger.info(f"Results DataFrame shape: {df.shape}")
                logger.info(f"Results DataFrame columns: {df.columns.tolist()}")

                # Log results for each question
                for idx, row in df.iterrows():
                    logger.info(f"--- Results for Question {idx + 1} ---")
                    logger.info(f"Question: {eval_questions[idx] if idx < len(eval_questions) else 'N/A'}")
                    logger.info(f"Expected Answer: {eval_answers[idx][:100] if idx < len(eval_answers) else 'N/A'}...")
                    if 'answer' in row:
                        logger.info(f"RAG Answer: {str(row['answer'])[:100]}...")
                    for col in df.columns:
                        if col in ['faithfulness', 'answer_relevancy', 'context_precision', 'answer_correctness']:
                            logger.info(f"{col}: {row[col]}")
                    logger.info("-" * 60)
        except Exception as e:
            logger.error(f"Error during RAGAS evaluation: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")

            # Provide more specific error messages
            error_msg = str(e)
            if "api" in error_msg.lower() and "key" in error_msg.lower():
                error_msg = "Google API key issue. Please check your GOOGLE_API_KEY environment variable."
            elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
                error_msg = "API quota exceeded. Please check your Google API usage limits."
            elif "timeout" in error_msg.lower():
                error_msg = "Request timeout. The evaluation might be taking too long - try with fewer questions."
            elif "json" in error_msg.lower() and "serializable" in error_msg.lower():
                error_msg = "Result serialization error. The evaluation completed but results couldn't be formatted properly."

            return jsonify({"error": f"RAGAS evaluation failed: {error_msg}"}), 500