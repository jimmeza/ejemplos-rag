import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from rag_helper.build_vector_db import (
                                        crear_dataset_evaluacion,
                                        crear_vector_store,
                                        evaluar_documentos_recuperados)


def main(args=None):
    #Configurar logging a archivo (sobreescribe el archivo en cada ejecución)
    nombre_archivo = Path(__file__).stem
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=f"logs/{nombre_archivo}.log", filemode="w", encoding="utf-8")

    load_dotenv(override=True)

    project_name = os.getenv("LANGCHAIN_PROJECT")
    print("Proyecto LangSmith:", project_name)
    COLLECTION_NAME = "futbol-peruano-hybrid-doc-chunks"
    COLLECTION_NAME_CUSTOM = "futbol-peruano-hybrid-markdown-chunks"
    
    #Vector stores para probar diferentes estrategias de chunking
    vs_doc_chunks = crear_vector_store(COLLECTION_NAME, 
                                      modo_recuperacion="hybrid",
                                      )

    vs_custom_chunks = crear_vector_store(COLLECTION_NAME_CUSTOM, 
                                      modo_recuperacion="hybrid",
                                      )

    querys = [
        "¿Cuál fue el club ganador de copa bicentenario 2019?",
        "¿En qué temporada ganó Juan Aurich su primer campeonato?",
        "¿Quien ganó la Supercopa 'Copa Federación' del 2012?, indicar el subcampeon.",
    ]

    respuestas_ok = [
        "El club Atlético Grau gano la copa Bicentenario del 2019",
        "Juan Aurich gano su primer campeonato en la temporada del año 2011.",
        "José Gálvez ganó la Supercopa “Copa Federación” del 2012, Juan Aurich fue el subcampeón.",
    ]

    llm = init_chat_model("openai/gpt-oss-20b", model_provider="groq", temperature=0.0)

    # Preparar datasets para probar diferentes estrategias de chunking
    dataset_doc_chunks = crear_dataset_evaluacion(vs_doc_chunks, 
                            llm, 
                            querys,
                            respuestas_ok,
                            max_docs_recuperados=3,
                            )
    dataset_custom_chunk = crear_dataset_evaluacion(vs_custom_chunks, 
                            llm, 
                            querys,
                            respuestas_ok,
                            max_docs_recuperados=3,
                            )

    print(f"Probando busqueda híbrida con chunking deafult en colección: {COLLECTION_NAME}")
    evaluacion_doc_chunks = evaluar_documentos_recuperados(dataset_doc_chunks, llm, imprimir_resultados=True)
    print(f"Probando busqueda hibrida con chunking personalizador en colección: {COLLECTION_NAME_CUSTOM}")
    evaluacion_custom_chunks = evaluar_documentos_recuperados(dataset_custom_chunk, llm, imprimir_resultados=True)


if __name__ == "__main__":
    main()

