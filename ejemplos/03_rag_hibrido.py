import logging
from pathlib import Path

from dotenv import load_dotenv

from rag_helper.build_vector_db import cargar_documentos, poblar_coleccion


def main(args=None):
    #Configurar logging a archivo (sobreescribe el archivo en cada ejecución)
    nombre_archivo = Path(__file__).stem
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=f"logs/{nombre_archivo}.log", filemode="w", encoding="utf-8")

    ruta_archivos = [
        "https://es.wikipedia.org/wiki/F%C3%BAtbol_en_el_Per%C3%BA",
        "https://es.wikipedia.org/wiki/Copas_nacionales_del_f%C3%BAtbol_peruano",
        "https://es.wikipedia.org/wiki/Primera_Divisi%C3%B3n_del_Per%C3%BA",
        #"./PDF/Primera_División_del_Perú.pdf",  #Procesar un PDF es mucho mas lento que un documento HTML
    ]
    load_dotenv(override=True)
    
    COLLECTION_NAME = "futbol-peruano-hybrid-doc-chunks"
    docs = cargar_documentos(ruta_archivos, tipo_exportacion="doc_chunks", max_chunk_token_size=1024)
    vector_store = poblar_coleccion(COLLECTION_NAME, docs, "hybrid", recrear_coleccion=True)

    CUSTOM_COLLECTION_NAME = "futbol-peruano-hybrid-markdown-chunks"
    docs_md = cargar_documentos(ruta_archivos, tipo_exportacion="custom_chunks", max_chunk_token_size=1024)
    vector_store_custom = poblar_coleccion(CUSTOM_COLLECTION_NAME, docs_md, "hybrid", recrear_coleccion=True)

if __name__ == "__main__":
    main()


