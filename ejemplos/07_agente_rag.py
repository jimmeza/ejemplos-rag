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
    #Iniciamos el modelo de lenguaje para el agente RAG, se puede usar el mismo modelo para el agente RAG y para el reranker por LLM, 
    #pero en esta evaluación usaremos un modelo más pequeño para el reranker y así reducir costos en esta etapa de evaluación del agente RAG.
    llm = init_chat_model("openai/gpt-oss-120b", model_provider="groq", temperature=0.0)

    #inicializamos la clase QueryRAG con el modelo de lenguaje, el vector store y sin reranker para la primera evaluación del agente RAG
    query_rag_hybrid = QueryRAG(model=llm, 
                                vector_store=vs_custom_chunks,
                                reranker=None,
                                )

    query = """Identifica las copas, campeonatos o juegos internacionales en las que participó la selección de Perú de mayores 
    y muestrame solo las 11 ocasiones en que ganó en una tabla indicando el año y la competencia."""

    print("Prueba de consulta con Agente RAG sin Reranker y respuesta resumen")
    query_rag_hybrid.consultar(query, max_docs_recuperados=5, max_busquedas=5, limite_recursion=15, imprimir_respuesta="resumen")
    print("\n","*"*100, "\n")
    
    #Asignamos reranker a la clase para evaluar el mismo agente RAG pero con reranker por modelo reranker ejecutado localmente
    query_rag_hybrid.reranker = "local"
    print("Prueba de consulta con Agente RAG usando Reranker con un modelo reranker local, sin filtrado por umbral de relevancia y respuesta total (con documentos recuperados)")
    query_rag_hybrid.consultar(query, max_docs_recuperados=5, max_busquedas=5, limite_recursion=15, imprimir_respuesta="total")
    print("\n","*"*100, "\n")
    
    #Asignamos reranker a la clase para evaluar el mismo agente RAG pero con reranker por LLM
    query_rag_hybrid.reranker = "llm"
    #Usaremos un llm más pequeño para el reranker y así reducir costos en esta etapa de evaluación del agente RAG, ya que el reranker se ejecuta varias veces durante la consulta.
    query_rag_hybrid.llm_reranker = init_chat_model("openai/gpt-oss-20b", model_provider="groq", temperature=0.0)
    print("Prueba de consulta con Agente RAG usando Reranker por LLM y filtrado por umbral de relevancia = 0.5, sin imprimir respuesta")
    respuesta = query_rag_hybrid.consultar(query, max_docs_recuperados=5, max_busquedas=5, limite_recursion=15, umbral=0.5)
    print("Respuesta del Agente (Manualmente impresa):\n", respuesta)

    # respuestas_ok = [
    #     """Las ocasiones en que la selección de Perú ganó en copas, campeonatos o juegos internacionales son:
    #     | Año | Competición ganada |
    #     |-----|--------------------|
    #     | 1938 | Juegos Bolivarianos |
    #     | 1939 | Campeonato Sudamericano (Copa América) |
    #     | 1947-1948 | Juegos Bolivarianos |
    #     | 1961 | Juegos Bolivarianos |
    #     | 1973 | Juegos Bolivarianos |
    #     | 1975 | Copa América |
    #     | 1981 | Juegos Bolivarianos |
    #     | 1999 | Copa Kirin (empate con Bélgica)|
    #     | 2000 | Copa Oro de la CONCACAF |
    #     | 2005 | Copa Kirin |
    #     | 2011 | Copa Kirin |
    #     """,
    # ]
    
if __name__ == "__main__":
    main()

