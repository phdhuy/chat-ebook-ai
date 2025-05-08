from elasticsearch import Elasticsearch
from config import ES_HOST, ES_API_KEY, ES_PORT, ES_USER, ES_PASS
import logging

logger = logging.getLogger(__name__)


def connect_elasticsearch():
    headers = {
        "Accept": "application/vnd.elasticsearch+json; compatible-with=8",
        "Content-Type": "application/vnd.elasticsearch+json; compatible-with=8"
    }
    try:
        logger.info(f"Connecting to Elasticsearch with headers: {headers}")
        if ES_HOST and ES_API_KEY:
            es = Elasticsearch(
                hosts=[ES_HOST],
                api_key=ES_API_KEY,
                headers=headers
            )
            logger.info("Connected to Elasticsearch Cloud")
        else:
            es_args = {
                "hosts": [{"host": ES_HOST, "port": int(ES_PORT), "scheme": "http"}],
                "headers": headers
            }
            if ES_USER and ES_PASS:
                es_args['basic_auth'] = (ES_USER, ES_PASS)
            es = Elasticsearch(**es_args)
            logger.info("Connected to Elasticsearch at %s:%s", ES_HOST, ES_PORT)

        logger.debug("Sending info request to verify connection")
        return es
    except Exception as e:
        logger.error(f"Unexpected error connecting to Elasticsearch: {str(e)}")
        raise

def get_embedding_dimension(es, index):
    try:
        mapping = es.indices.get_mapping(index=index)
        properties = mapping[index]['mappings']['properties']
        if 'embedding' in properties and properties['embedding']['type'] == 'dense_vector':
            return properties['embedding']['dims']
    except Exception as e:
        logger.error("Failed to get embedding dimension: %s", e)
    return None