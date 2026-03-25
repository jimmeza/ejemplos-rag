import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from rag_helper.build_vector_db import crear_vector_store
from rag_helper.query_rag import QueryRAG


def main(args=None):
    #Configurar logging a archivo (sobreescribe el archivo en cada ejecución)
    nombre_archivo = Path(__file__).stem
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=f"logs/{nombre_archivo}.log", filemode="w", encoding="utf-8")

    load_dotenv(override=True)

    project_name = os.getenv("LANGCHAIN_PROJECT")
    print("Proyecto LangSmith:", project_name)
    COLLECTION_NAME = "futbol-peruano-hybrid-markdown-chunks"
    
    #Vector stores para RAG de Agente IA
    vs_custom_chunks = crear_vector_store(COLLECTION_NAME)

    querys = [
        "¿Cuál es el primer club ganador de la Copa Bicentenario y en que año se fundo el club?",
        "¿Cuantas veces fue Perú a los juegos olimpicos, en que fase quedó y que puesto obtuvo?",
        "¿Cuál fue el club ganador de copa bicentenario 2019?",
        "¿En qué temporada ganó Juan Aurich su primer campeonato?",
        "¿Quien ganó la Supercopa 'Copa Federación' del 2012?, indicar el subcampeon.",
    ]

    respuestas_ok = [
        "El primner club ganador de la Copa Bicentenario fue el Atlético Grau en el año 2019 y se fundo el 5 de junio de 1919.",
        "Perú participo en las olimpiadas en dos ocasiones, en Berlín 1936 (llegando hasta los cuartos de final y se retiró después del controvertido partido contra Austria, sin obtener un puesto) y en Roma 1960 (quedó en la fase inicial o de grupos y ocupó el 11° lugar).",
        "El club Atlético Grau gano la copa Bicentenario del 2019",
        "Juan Aurich gano su primer campeonato en la temporada del año 2011.",
        "José Gálvez ganó la Supercopa “Copa Federación” del 2012, Juan Aurich fue el subcampeón.",
    ]

    llm = init_chat_model("openai/gpt-oss-120b", model_provider="groq", temperature=0.0)

    query_rag_hybrid = QueryRAG(llm, 
                                vs_custom_chunks,
                                reranker=None, #Sin reranker para la primera evaluación del agente RAG
                                )
    print("Pruebas de Agente RAG sin Reranker")
    query_rag_hybrid.evaluar_agente_rag(COLLECTION_NAME, querys, respuestas_ok, nivel_detalle="total", imprimir_respuestas="resumen")
    print("\n","*"*100, "\n")
    #Asignamos reranker a la clase para evaluar el mismo agente RAG pero con reranker por LLM
    query_rag_hybrid.reranker = "llm"
    print("Pruebas de Agente RAG usando Reranker por LLM sin filtrado por umbral de relevancia")
    query_rag_hybrid.evaluar_agente_rag(COLLECTION_NAME, querys, respuestas_ok, nivel_detalle="total", imprimir_respuestas="resumen")
    print("\n","*"*100, "\n")
    print("Pruebas de Agente RAG usando Reranker por LLM y filtrado por umbral de relevancia = 0.5")
    query_rag_hybrid.evaluar_agente_rag(COLLECTION_NAME, querys, respuestas_ok, nivel_detalle="total", umbral=0.5, imprimir_respuestas="resumen")
    

if __name__ == "__main__":
    main()

