import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from rag_helper.build_vector_db import (OLLAMA_DEFAULT_EMBEDDING_MODEL,
                                        OLLAMA_SMALL_EMBEDDING_MODEL,
                                        build_embeddings,
                                        crear_dataset_evaluacion,
                                        crear_vector_store,
                                        evaluar_documentos_recuperados)

def main(args=None):
    #Configurar logging a archivo (sobreescribe el archivo en cada ejecución)
    nombre_archivo = Path(__file__).stem
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=f"logs/{nombre_archivo}.log", filemode="w", encoding="utf-8")

    load_dotenv(override=True)

    COLLECTION_NAME_SMALL = "futbol-peruano-denso-small"
    COLLECTION_NAME = "futbol-peruano-denso-default"
    
    #Colección con vector de 384 dimensiones (Embeddings small)
    print("Colección con vector denso de 384 dimensiones:", COLLECTION_NAME_SMALL)
    embedding_small = build_embeddings(OLLAMA_SMALL_EMBEDDING_MODEL)
    vector_store_small = crear_vector_store(COLLECTION_NAME_SMALL, 
                                      embeddings = embedding_small,
                                      modo_recuperacion="dense",
                                      )
    #Colección con vector de 1024 dimensiones (Embeddings default)
    print("Colección con vector denso de 1024 dimensiones:", COLLECTION_NAME)
    embedding_default = build_embeddings(OLLAMA_DEFAULT_EMBEDDING_MODEL)
    vector_store_default = crear_vector_store(COLLECTION_NAME, 
                                      embeddings = embedding_default,
                                      modo_recuperacion="dense",
                                      )

    querys = [
    "¿Cuando fue el primer encuentro de fútbol en el Perú?",
    "club ganador de copa bicentenario 2019",
    "primer título internacional de selección nacional de Perú"
    ]

    respuestas_ok = [
    "el primer registro de un partido de fútbol en el Perú, corresponde al domingo 7 de agosto de 1892",
    "El club Atlético Grau gano la copa Bicentenario del 2019",
    "El primer título internacional de la selección de Perú fue en los Juegos Bolivarianos de 1938"
    ]
    
    llm = init_chat_model("openai/gpt-oss-20b", model_provider="groq", temperature=0.0)

    dataset_small = crear_dataset_evaluacion(vector_store_small, 
                            llm, 
                            querys,
                            respuestas_ok,
                            max_docs_recuperados=5,
                            )
    dataset_default = crear_dataset_evaluacion(vector_store_default, 
                            llm, 
                            querys,
                            respuestas_ok,
                            max_docs_recuperados=5,
                            )

    print(f"Probando colección de 384 dimensiones: {COLLECTION_NAME_SMALL}")
    evaluacion_small = evaluar_documentos_recuperados(dataset_small, llm, imprimir_resultados=True)
    print("="*50, "\n", f"Probando colección de 1024 dimensiones: {COLLECTION_NAME}")
    evaluacion_default = evaluar_documentos_recuperados(dataset_default, llm, imprimir_resultados=True)

if __name__ == "__main__":
    main()
