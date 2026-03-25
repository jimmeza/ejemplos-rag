from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenReranker:
    def __init__(self, model_name_or_path="Qwen/Qwen3-Reranker-0.6B", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).eval()
        
        if device:
            self.model.to(device)
        elif torch.cuda.is_available():
            self.model.to("cuda")
        elif torch.backends.mps.is_available():
            self.model.to("mps")
            
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("si")
        self.max_length = 32*1024 #8192
        
        self.prefix = "<|im_start|>system\nDetermine si el documento cumple los requisitos según la consulta y la instrucción proporcionada. Tenga en cuenta que la respuesta solo puede ser \"si\" o \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

    def _format_instruction(self, instruction, query, doc):
        """
        Formats the instruction for the model by adding the query and document to it.
        Args:
            instruction (str): The instruction to format.
            query (str): The query to be used in the instruction.
        """
        if instruction is None:
            instruction = 'Dada una consulta, recuperar pasajes relevantes que respondan a la consulta'
        return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
            instruction=instruction, query=query, doc=doc
        )

    def _process_inputs(self, pairs):
        """
        Processes the input pairs by tokenizing them and adding the prefix and suffix tokens.
        Args:
            pairs (list of tuples): A list of tuples where each tuple contains a query and a document.
        Returns:
            dict: The processed inputs.
        """
        inputs = self.tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, 
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs

    @torch.no_grad()
    def _compute_logits(self, inputs):
        """
        Computes the logits for the given inputs.
        """
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    def rerank(self, pairs: List[Tuple[str, str]], task:str|None=None) -> List[Tuple[str, str, float]]:
        """
        Reordena un conjunto de pares consulta-documento.
        Args:
            pairs: Una lista de tuplas (consulta, documento).
            task: La instrucción de la tarea.
        """
        formatted_pairs = [self._format_instruction(task, q, d) for q, d in pairs]
        inputs = self._process_inputs(formatted_pairs)
        scores = self._compute_logits(inputs)
        
        return [(q, d, float(s)) for (q, d), s in zip(pairs, scores)]

def main(args=None):
    # Ejemplo de uso del reranker ejecutando en local usando el modelo reranker "Qwen3-Reranker-0.6B"
    import time
    start_time = time.time()

    reranker = QwenReranker()
    
    task = None
    queries = "¿Cuál es la capital de Francia?"
    documents = [
        "París es la capital de Francia y una ciudad hermosa.",
        "El río Sena fluye a través de París.",
        "Berlín es la capital de Alemania.",
        "Los beneficios de la energía solar incluyen la reducción de la huella de carbono y el ahorro de costos a largo plazo."
    ]
    pairs=[(queries, doc) for doc in documents]
    results = reranker.rerank(pairs=pairs, task=task)
    for q, d, s in results:
        print(f"Query: {q}")
        print(f"Doc: {d}")
        print(f"Score: {s:.2f}")
        print("-" * 50)
    
    print(f"Tiempo de ejecución: {time.time() - start_time:.2f}")    

# --- Código para Pruebas ---
if __name__ == "__main__":
    main()