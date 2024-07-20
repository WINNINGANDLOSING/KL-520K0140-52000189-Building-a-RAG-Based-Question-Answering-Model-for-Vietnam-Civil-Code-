import base64
import gc
import random
import tempfile
import time
import uuid

from llama_index.llms.ollama import Ollama
from llama_index.core.schema import QueryBundle


import RAG_components as rag
from dotenv import load_dotenv
import os

load_dotenv()

import streamlit as st

os.environ['ACTIVELOOP_TOKEN'] = "eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJpZCI6Imh1bnRlciIsImFwaV9rZXkiOiJ6ejh5dG1sZURjNWkwc3g3NldNRjF5NWNrUWNIMnlfenR2VHV6MERmM3ppMVAifQ."

def init():
    if "id" not in st.session_state:
        st.session_state.id = uuid.uuid4()
        st.session_state.file_cache = {}

    session_id = st.session_state.id
    client = None

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()


def answer(query_str, retrievers, llm, reranker, generator_prompt):
    queries = rag.generate_queries(llm, query_str)
    
    nodes = []
    for retriever in retrievers:
        for q in queries:
            retrieved_nodes = retriever.retrieve(q)
            for n in retrieved_nodes:
                nodes.append(n)
        nodes.append(retriever.retrieve(query_str)[0])
    
    nodes = [node for node in nodes if node.score >= 0.75]
    final_nodes = reranker.postprocess_nodes(
        nodes, QueryBundle(query_str)
    )

    context = "\n\n".join([node.get_content() for node in final_nodes])
    response = llm.complete(generator_prompt.format(context_str=context, query_str=query_str))
    return response


def main():
    ## ================================================
    init()


    ## =======================
    #### reranker
    from llama_index.postprocessor.cohere_rerank import CohereRerank

    cohere_rerank = CohereRerank(model="rerank-multilingual-v2.0", api_key=os.getenv('COHERE_API_KEY'), top_n=3)  # remain top 3 relevant

    ## set up query engine
    embed_model = rag.get_embedding_model()

    from llama_index.core import Settings
    from llama_index.llms.gemini import Gemini
    Settings.embed_model = embed_model

    llm = Gemini()
    # llm = rag.get_llm()
    # llm=Ollama(model="llama3-chatqa", request_timeout=120.0)
    Settings.llm = llm 
    vector_store = rag.get_vector_database("hunter", "Vietnamese-law-RAG")
    index = rag.get_index(vector_store=vector_store)
    # query_engine = index.as_query_engine(similarity_top_k=3, streaming=True)
    # vector_store = get_vector_database("hunter", "Vietnamese-law-RAG")
    index = rag.get_index(vector_store=vector_store)
    vector_retriever = index.as_retriever(similarity_top_k=3)
    bm25_retriever = rag.get_bm25_retriever(index)
    retrievers = [vector_retriever, bm25_retriever]

    gen_queries_prompt = rag.prompt_for_generating_query
    generator_prompt = rag.prompt_

    ## =======================

    col1, col2 = st.columns([6, 1])

    with col1:
        st.header(f"Legal Chat using Llama-3")

    with col2:
        st.button("Clear ↺", on_click=reset_chat)

    # Initialize chat history
    if "messages" not in st.session_state:
        reset_chat()

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What's up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # # Simulate stream of response with milliseconds delay
            # streaming_response = query_engine.query(prompt)
            
            # for chunk in streaming_response.response_gen:
            #     full_response += chunk
            #     message_placeholder.markdown(full_response + "▌")
            response = answer(prompt, retrievers, llm, cohere_rerank, generator_prompt)

            full_response = response.text # query_engine.query(prompt)

            message_placeholder.markdown(full_response)
            # st.session_state.context = ctx

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()