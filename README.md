# Ejemplos de RAG



## Ejemplo NO Code
Estos ejemplos están basados en el artículo "[¿Pierdes Tiempo Buscando en tus Documentos o Webs? Tal Vez Necesitas una Base de Conocimiento IA (y Saber Cómo Crear una)](https://www.linkedin.com/pulse/pierdes-tiempo-buscando-en-tus-documentos-o-webs-tal-vez-jim-meza-bmrce/)"

Archivos JSON con los flujos de trabajo de ejemplo en [LangFlow](https://github.com/jimmeza/ejemplos-rag/tree/main/LangFlow)

---

## Ejemplos de Mejoras para RAG
Estos ejemplos están basados en el artículo "[De respuestas mediocres a precisas: Mejorando la recuperación de tu RAG en tu base de conocimiento IA](https://www.linkedin.com/pulse//)"

Los módulos core están en la carpeta [rag_helper](https://github.com/jimmeza/ejemplos-rag/tree/main/rag_helper), algunos tienen la función main() con código de prueba de su mismo módulo.

Los módulos de los ejemplos están en la carpeta [ejemplos](https://github.com/jimmeza/ejemplos-rag/tree/main/ejemplos), al ejecutarlos dejan un archivo .log en la carpeta logs y se deben ejecutar como módulos de python (con el argumento -m y la ruta 'paquete'.'modulo'), por ejemplo para ejecutar el primer ejemplo del archivo '01_rag_embeddings_diferentes.py', se debe ejecutar desde la raiz del proyecto el comando:

```
py -m ejemplos.01_rag_embeddings_diferentes
```

Se puede probar con diferentes modelos LLM para el agente RAG y como reranker:
Como LLM se puede usar cualquier modelo disponibles con init_chat_model de LangChain, usando Ollama o Llama.cpp(en local), o usando las APIs de Gemini, OpenAI, Groq u OpenRouter (se necesita su API Key correspondiente en archivo .env)

Para el agente RAG, se recomienda usar un modelo potente para que pueda entender la pregunta, realizar las búsquedas iterativas y generar una respuesta precisa basada en los documentos recuperados (en los ejemplos usamos gpt-oss-120b). 

Para el reranker, se puede usar un modelo más pequeño pero que sea capaz de evaluar la relevancia de los documentos recuperados para la pregunta dada (en los ejemplos usamos gpt-oss-20b).

Para pruebas rápidas se puede usar un modelo pequeño para el agente RAG y ningun reranker, o un LLM más pequeño para el reranker, pero para obtener mejores resultados se recomienda usar modelos potentes para ambos (en los ejemplos usamos gpt-oss-20b).

```
#Ejemplos de como inicializar el LLM con cada proveedor probado

#Gemini
llm = init_chat_model("gemini-3.1-flash-lite-preview", model_provider="google_genai", temperature=0.0)

#Ollama
llm = init_chat_model("qwen3.5:9b", model_provider="ollama", temperature=0.0)

#Groq
llm = init_chat_model("openai/gpt-oss-120b", model_provider="groq", temperature=0.0)

#OpenAI
llm = init_chat_model("gpt-5-mini", model_provider="openai", temperature=0.0)

#OpenRouter
llm = init_chat_model("minimax/minimax-m2.5:free", model_provider="openai", 
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.0)

#Llama.cpp local (se necesita tener el modelo descargado y configurado en local (llama-server) y usar el base_url con el puerto correspondiente)
llm = init_chat_model(model="local", 
                    model_provider="openai", 
                    base_url="http://localhost:8080/v1",
                    api_key="no-necesita",
                    temperature=0.0,
                    )

```

Tambíen se debe preparar el archivo **.env** en base al archivo **.env.example** con sus respectivas API Keys.

Para los demás requisitos se debe revisar el artículo "[De respuestas mediocres a precisas: Mejorando la recuperación de tu RAG en tu base de conocimiento IA](https://www.linkedin.com/pulse//)".

