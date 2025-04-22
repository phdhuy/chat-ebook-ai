import PyPDF2
from pdf2image import convert_from_path
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import nltk
from elasticsearch.helpers import bulk
import logging

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

def process_pdf(filepath, es, embedder, ES_INDEX):
    text = ""
    try:
        with open(filepath, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        logger.info("Text extracted using PyPDF2")
    except Exception as e:
        logger.warning(f"PyPDF2 extraction failed: {e}")
    if not text.strip():
        images = convert_from_path(filepath, dpi=300)
        for img in images:
            img = preprocess_image_for_ocr(img)
            text += pytesseract.image_to_string(img, config='--psm 6') + "\n"
        logger.info("Text extracted using OCR")
    if not text.strip():
        raise ValueError("No extractable text found in PDF.")
    text = ' '.join(text.split())
    sentences = nltk.sent_tokenize(text)
    chunk_size, overlap = 5, 2
    chunks = []
    for i in range(0, len(sentences), chunk_size - overlap):
        end = min(i + chunk_size, len(sentences))
        chunk = ' '.join(sentences[i:end])
        if chunk:
            chunks.append(chunk)
        if end == len(sentences): break
    logger.info(f"Created {len(chunks)} chunks")
    embeds = embedder.encode(chunks, convert_to_tensor=False, normalize_embeddings=True)
    dimension = len(embeds[0])
    if not es.indices.exists(index=ES_INDEX):
        mapping = {
            "mappings": {
                "properties": {
                    "chunk": {"type": "text"},
                    "embedding": {"type": "dense_vector", "dims": dimension}
                }
            }
        }
        es.indices.create(index=ES_INDEX, body=mapping)
        logger.info(f"Created Elasticsearch index '{ES_INDEX}' with dims={dimension}")
    actions = []
    for i, vec in enumerate(embeds):
        actions.append({
            "_index": ES_INDEX,
            "_id": i,
            "_source": {"chunk": chunks[i], "embedding": vec.tolist()}
        })
    bulk(es, actions)
    logger.info(f"Indexed {len(actions)} documents into '{ES_INDEX}'")
    return len(chunks)  # Return number of chunks for the upload response