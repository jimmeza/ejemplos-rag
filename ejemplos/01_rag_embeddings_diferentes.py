import logging
from pathlib import Path

from dotenv import load_dotenv

from rag_helper.build_vector_db import (HF_SMALL_EMBEDDING_MODEL,
                                        MAX_TOKENS_SMALL_EMBEDDING,
                                        OLLAMA_SMALL_EMBEDDING_MODEL,
                                        SMALL_VECTOR_SIZE, cargar_documentos,
                                        poblar_coleccion)


def main(args=None):
    #Configurar logging a archivo (sobreescribe el archivo en cada ejecución)
    nombre_archivo = Path(__file__).stem
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=f"logs/{nombre_archivo}.log", filemode="w", encoding="utf-8")

    ruta_archivos = [
        "https://es.wikipedia.org/wiki/F%C3%BAtbol_en_el_Per%C3%BA",
        "https://es.wikipedia.org/wiki/Copas_nacionales_del_f%C3%BAtbol_peruano",
        "https://es.wikipedia.org/wiki/Primera_Divisi%C3%B3n_del_Per%C3%BA",
    ]
    load_dotenv(override=True)
    
    COLLECTION_NAME = "futbol-peruano-denso-small"
    COLLECTION_NAME_LARGE = "futbol-peruano-denso-default"
    #Cargamos los documentos usando la conversion default de Docling y con el tamaño de chunk pequeño de 256 Tokens
    docs = cargar_documentos(
        ruta_archivos, 
        tokenizer_hf_model = HF_SMALL_EMBEDDING_MODEL,
        tipo_exportacion="custom_chunks", 
        max_chunk_token_size=MAX_TOKENS_SMALL_EMBEDDING,
    )

    
    #Con los mismos chunks pequeños de no más de 256 Tokens, creamos dos colecciones con diferentes embeddings
    #Una con embeddings de 1024 dimensiones (el default para la función)
    poblar_coleccion(COLLECTION_NAME_LARGE, docs, "dense", recrear_coleccion=True)
    #Otra con embeddings de 384 dimensiones (se debe indicar el modelo de Ollama)
    poblar_coleccion(COLLECTION_NAME, docs, "dense", recrear_coleccion=True,
                                        embedding_model= OLLAMA_SMALL_EMBEDDING_MODEL, 
                                        dimensiones_vector_denso=SMALL_VECTOR_SIZE)
  
if __name__ == "__main__":
    main()


