import fitz
import nltk
from elasticsearch.helpers import bulk
import logging
from PIL import Image, ImageEnhance, ImageFilter
from pdf2image import convert_from_path
import pytesseract
import uuid

logger = logging.getLogger(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'

def preprocess_image_for_ocr(image):
    image = image.convert('L')
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)
    image = image.filter(ImageFilter.SHARPEN)
    image = image.resize((int(image.width * 1.5), int(image.height * 1.5)), Image.LANCZOS)
    return image

def process_pdf(filepath, es, embedder, ES_INDEX, conversation_id, return_chunks=False):
    text_by_page = []
    try:
        doc = fitz.open(filepath)
        for page_num in range(len(doc)):
            page = doc[page_num]
            extracted = page.get_text()
            if extracted.strip():
                text_by_page.append({"page": page_num + 1, "text": extracted})
        doc.close()
        logger.info("Text extracted using PyMuPDF")
    except Exception as e:
        logger.warning(f"PyMuPDF extraction failed: {e}")

    if not text_by_page:
        logger.info("No text extracted with PyMuPDF, attempting OCR")
        images = convert_from_path(filepath, dpi=300)
        for i, img in enumerate(images):
            img = preprocess_image_for_ocr(img)
            text = pytesseract.image_to_string(img, config='--psm 6')
            if text.strip():
                text_by_page.append({"page": i + 1, "text": text})
        logger.info("Text extracted using OCR")

    if not text_by_page:
        raise ValueError("No extractable text found in PDF.")

    chunk_size, overlap = 18, 1
    chunks = []
    for page_data in text_by_page:
        page_num = page_data["page"]
        page_text = ' '.join(page_data["text"].split())
        sentences = nltk.sent_tokenize(page_text)
        for i in range(0, len(sentences), chunk_size - overlap):
            end = min(i + chunk_size, len(sentences))
            chunk_text = ' '.join(sentences[i:end])
            if chunk_text:
                chunks.append({"text": chunk_text, "page": page_num})
            if end == len(sentences):
                break
    logger.info(f"Created {len(chunks)} chunks")

    batch_size = 50
    embeds = []
    for i in range(0, len(chunks), batch_size):
        batch_texts = [chunk["text"] for chunk in chunks[i:i + batch_size]]
        batch_embeds = embedder.encode(batch_texts, convert_to_tensor=False, normalize_embeddings=True)
        embeds.extend(batch_embeds)
    dimension = len(embeds[0])

    if not es.indices.exists(index=ES_INDEX):
        mapping = {
            "mappings": {
                "properties": {
                    "chunk": {"type": "text"},
                    "page": {"type": "integer"},
                    "conversation_id": {"type": "keyword"},
                    "embedding": {"type": "dense_vector", "dims": dimension}
                }
            }
        }
        es.indices.create(index=ES_INDEX, body=mapping)
        logger.info(f"Created Elasticsearch index '{ES_INDEX}' with dims={dimension}")

    actions = []
    for i, vec in enumerate(embeds):
        unique_id = str(uuid.uuid4())
        actions.append({
            "_index": ES_INDEX,
            "_id": unique_id,
            "_source": {
                "chunk": chunks[i]["text"],
                "page": chunks[i]["page"],
                "conversation_id": conversation_id,
                "embedding": vec.tolist()
            }
        })
    bulk_size = 200
    for i in range(0, len(actions), bulk_size):
        bulk(es, actions[i:i + bulk_size])
    logger.info(f"Indexed {len(actions)} documents into '{ES_INDEX}' with conversation_id={conversation_id}")

    if return_chunks:
        return chunks
    return len(chunks)