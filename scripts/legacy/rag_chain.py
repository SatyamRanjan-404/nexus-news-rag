import os
from dotenv import load_dotenv
from retriever import retrieve_multi_source_context
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def get_rag_stream_and_sources(user_query, source_filter=None,days_filter=None):
    """
    Accepts the user query and an optional source URL string filter,
    routing both directly down into the database layer.
    """
    # Pass the source_filter directly to our retrieval call
    retrieved_data = retrieve_multi_source_context(user_query, top_k=4, source_filter=source_filter)

    if not retrieved_data or not retrieved_data['context'].strip():
        return None, []

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2
    )

    system_rules = """
    You are an expert, descriptive intelligence analyst. 
    You are analyzing multiple incoming live news broadcast feeds regarding the user's inquiry.
    
    YOUR INSTRUCTIONS:
    1. Provide a thorough, comprehensive, and descriptive answer synthesizing the provided news sources.
    2. Do not give a simple 1-sentence answer. Expand on implications, timelines, and figures if mentioned.
    3. Rely strictly on the text provided under the 'NEWS CONTEXT' block.
    4. If none of the sources actually answer the user's specific question, reply exactly with: 
       "I cannot find explicit details about that topic in today's collected news reports."

    NEWS CONTEXT:
    {context}
    """

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_rules),
        ("user", "{question}")
    ])

    rag_chain = prompt_template | llm | StrOutputParser()

    stream_generator = rag_chain.stream({
        "context": retrieved_data['context'],
        "question": user_query
    })

    return stream_generator, retrieved_data['citations']