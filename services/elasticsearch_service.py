from elasticsearch import Elasticsearch
from config import ES_HOST, ES_API_KEY, ES_INDEX, ES_PORT, ES_USER, ES_PASS
import logging

logger = logging.getLogger(__name__)

def connect_elasticsearch():
    if ES_HOST and ES_API_KEY:
        es = Elasticsearch(hosts=ES_HOST, api_key=ES_API_KEY)
        logger.info("Connected to Elasticsearch Cloud")
    else:
        es_args = {"hosts": [{"host": ES_HOST, "port": ES_PORT}]}
        if ES_USER and ES_PASS:
            es_args['http_auth'] = (ES_USER, ES_PASS)
        es = Elasticsearch(**es_args)
        logger.info("Connected to Elasticsearch at %s:%s", ES_HOST, ES_PORT)
    return es

def get_embedding_dimension(es, index):
    try:
        mapping = es.indices.get_mapping(index=index)
        properties = mapping[index]['mappings']['properties']
        if 'embedding' in properties and properties['embedding']['type'] == 'dense_vector':
            return properties['embedding']['dims']
    except Exception as e:
        logger.error("Failed to get embedding dimension: %s", e)
    return None