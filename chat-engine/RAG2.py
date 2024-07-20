from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.schema import QueryBundle
from llama_index.llms.gemini import Gemini

import os

from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.postprocessor.cohere_rerank import CohereRerank

from llama_index.llms.ollama import Ollama
from llama_index.core.settings import Settings

from dotenv import load_dotenv

from prompts import gen_qa_prompt, gen_rag_answer

load_dotenv()

class RAG:
    def __init__(
        self,
        model_name="vistral",
        embedding_model="/teamspace/studios/this_studio/bge-small-en-v1.5",
        dataset_name="Vietnamese-law-RAG"
    ):
        self.model = Ollama(model=model_name, request_timeout=120.0)
        self.embed_model = HuggingFaceEmbedding(model_name=embedding_model)
        Settings.embed_model = self.embed_model

        ## get vector store 
        self.activeloop_id = "hunter"
        self.dataset_name = dataset_name
        self.dataset_path = f"hub://{self.activeloop_id}/{self.dataset_name}"

        self.vector_store = DeepLakeVectorStore(
            dataset_path=self.dataset_path,
            overwrite=False,
        )

        self.index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)

        ## get retrievers
        self.vector_retriever = self.index.as_retriever(
            similarity_top_k=5
        )
        # bm25 retriever
        source_nodes = self.index.as_retriever(similarity_top_k=200000).retrieve("test")
        nodes = [x.node for x in source_nodes]
        self.bm25 = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)

        self.retrievers = [self.vector_retriever, self.bm25]

        ## metadata replacement and reranker
        self.replacement = MetadataReplacementPostProcessor(target_metadata_key="window")
        self.cohere_rerank = CohereRerank(model="rerank-multilingual-v2.0", api_key=os.getenv('COHERE_API_KEY'), top_n=3)  # remain top 3 relevant

    
    def generate_queries(self, query_str, num_queries=3, llm=Gemini(model_name="models/gemini-pro")):
        prompt = gen_qa_prompt.format(
            num_queries=num_queries, query=query_str
        )
        response = llm.complete(prompt)
        queries = response.text.split("\n")
        return queries
    
    def answer(self, query_str):
        # generate_queries
        queries = self.generate_queries(query_str)

        ## get relevant nodes
        nodes = []
        for retriever in self.retrievers:
            # Retrieve nodes for both queries and the original query
            for query in queries + [query_str]:
                retrieved_nodes = retriever.retrieve(query)
                nodes.extend([node for node in retrieved_nodes if node.score >= 0.7])
        
        # replace metadata
        nodes = self.replacement.postprocess_nodes(nodes)

        # rerank
        final_nodes = self.cohere_rerank.postprocess_nodes(
            nodes, QueryBundle(query_str)
        )

        context = "\n".join([node.get_content() for node in final_nodes])

        response = self.model.complete(
            gen_rag_answer.format(query_str=query_str, context_str=context)
        )

        return response

if __name__ == "__main__":
    rag = RAG(
        model_name="vistral"    # vistral, phogpt, vinallama
    )
    
    print(rag.answer("Người thành niên là từ bao nhiêu tuổi ?"))
