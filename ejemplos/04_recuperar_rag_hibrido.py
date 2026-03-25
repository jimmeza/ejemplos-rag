import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from rag_helper.build_vector_db import (crear_dataset_evaluacion,
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
    
    print("Colección:", COLLECTION_NAME)
    vector_store_hybrid = crear_vector_store(COLLECTION_NAME, 
                                      modo_recuperacion="hybrid",
                                      )

    vector_store_dense = crear_vector_store(COLLECTION_NAME, 
                                      modo_recuperacion="dense",
                                      )

    querys = [
    "¿Cuando fue el primer encuentro de fútbol en el Perú?",
    "¿Qué nos explica Jorge Basadre sobre el fútbol peruano?",
    "primer título internacional de selección nacional de Perú",
    ]

    respuestas_ok = [
    "el primer registro de un partido de fútbol en el Perú, corresponde al domingo 7 de agosto de 1892",
    "Jorge Basadre explica que el primer registro de un partido de fútbol en el Perú ocurrió el domingo 7 de agosto 1892, cuando ingleses y peruanos jugaron representando al Callao y a Lima. El club Lima Cricket and Lawn Tennis(el primer club del país) organizó encuentros en el campo Santa Sofía de su propiedad. La guerra del Pacífico provocó la destrucción de varias ciudades costeras, incluida Lima, lo que detuvo temporalmente la difusión del fútbol y otros deportes en el país.",
    "El primer título internacional de la selección de Perú fue en los Juegos Bolivarianos de 1938",
    ]
    
    llm = init_chat_model("openai/gpt-oss-20b", model_provider="groq", temperature=0.0)

    dataset_hybrid = crear_dataset_evaluacion(vector_store_hybrid, 
                            llm, 
                            querys,
                            respuestas_ok,
                            max_docs_recuperados=3,
                            )
    dataset_dense = crear_dataset_evaluacion(vector_store_dense, 
                            llm, 
                            querys,
                            respuestas_ok,
                            max_docs_recuperados=3,
                            )

    print(f"Probando busqueda semantica en colección: {COLLECTION_NAME}")
    evaluacion_semantica = evaluar_documentos_recuperados(dataset_dense, llm, imprimir_resultados=True)
    print(f"Probando busqueda hibrida en colección: {COLLECTION_NAME}")
    evaluacion_hibrida = evaluar_documentos_recuperados(dataset_hybrid, llm, imprimir_resultados=True)


if __name__ == "__main__":
    main()

