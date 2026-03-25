import logging
import os
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
        # "https://es.wikipedia.org/wiki/Copas_nacionales_del_f%C3%BAtbol_peruano",
        # "https://es.wikipedia.org/wiki/Primera_Divisi%C3%B3n_del_Per%C3%BA",
    ]
    load_dotenv(override=True)
    
    #Cargamos los documentos usando la conversion default de Docling y con el tamaño de chunk pequeño de 256 Tokens
    docs = cargar_documentos(
        ruta_archivos, 
        tipo_exportacion="custom_chunks", 
        max_chunk_token_size=1024,
    )

    ruta_archivo = "./temp/documento.md"
    os.makedirs(os.path.dirname(ruta_archivo), exist_ok=True)
    f = open(ruta_archivo, "w", encoding="utf-8")
    
    for doc in docs:
        f.write(f"---\n")
        f.write(f"source: {doc.metadata['source']}\n")
        f.write(f"page_content:\n{doc.page_content}\n")
    
    f.close()
    
    

if __name__ == "__main__":
    main()


