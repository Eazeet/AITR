import os
from langchain_core.tools import tool
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS

from file_encodings import (encode_restoration_chapter, 
                    encode_preliminary_inspection_chapter, encode_book)

from helper_functions import (
    num_tokens_from_string, replace_t_with_space, replace_double_lines_with_one_line, 
    split_into_chapters, analyse_metric_results, escape_quotes, text_wrap, 
    extract_book_quotes_as_documents, get_user_input
)

from some_knowledge_base_functions import create_chapter_summary

guides_path="pdf_assets/IICRC.pdf"
os.environ["OPENAI_API_KEY"] = ""
llm = ChatOpenAI(model="gpt-4-turbo-preview")

chapters = split_into_chapters(guides_path, start_page=89)
chapters = replace_t_with_space(chapters)

preliminary_inspection_chapter = [chapters[9]]
preliminary_inspection_chapter2 = chapters[9]
preliminary_inspection_summary = create_chapter_summary(preliminary_inspection_chapter2)
restoration_chapter = [chapters[12]]
@tool
def categorize_class(query: str) -> str:
    """
    Categorizes the given information about building damages into a specific category (Fire, Water, or General).
    
    Args:
        query (str): The description of building damage to be categorized.
        
    Returns:
        dict: A dictionary containing the category of the damage.
    """
    prompt = ChatPromptTemplate.from_template(
        "Categorize the following information about building damages into one of these categories: "
        "Fire, Water, General. Query: {query}"
    )
    chain = prompt | ChatOpenAI(temperature=0)
    category = chain.invoke({"query": query}).content
    return {"category": category}


@tool
def create_objectives(query: str) -> str:
    """
    Creates objectives for restoring building damages based on a specific category (Fire, Water, or General).
    
    Args:
        query (str): The description of building damage to generate restoration objectives.
        
    Returns:
        dict: A dictionary containing the objectives for restoring the damages.
    """
    prompt = ChatPromptTemplate.from_template(
        "Create objectives for restoring building damages categorized as: Fire, Water, General. Query: {query}"
    )
    chain = prompt | ChatOpenAI(temperature=0)
    objectives = chain.invoke({"query": query}).content
    return {"objectives": objectives}


@tool
def book_lookup(query: str) -> str:
    """
    Looks up relevant content in a pre-chunked book stored in FAISS based on the given query.
    
    Args:
        query (str): The query to search for in the book chunks.
        
    Returns:
        str: The most relevant content found in the book based on the query.
    """
    vector_store_path = "chunks_vector_store"
    if os.path.exists(vector_store_path):
        embeddings = OpenAIEmbeddings()
        chunks_vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    else:
        chunks_vector_store = encode_book(guides_path, chunk_size=10000, chunk_overlap=200)
        chunks_vector_store.save_local(vector_store_path)

    chunks_query_retriever = chunks_vector_store.as_retriever(search_kwargs={"k": 1})
    search_results = chunks_query_retriever.invoke(query)
    if search_results:
        return search_results[0].page_content  
    else:
        return "No relevant content found."


@tool
def preliminary_chapter_lookup(query: str) -> str:
    """
    Looks up relevant content in a pre-chunked book stored in FAISS based on the given query.
    
    Args:
        query (str): The query to search for in the book chunks.
        
    Returns:
        str: The most relevant content found in the book based on the query.
    """
    vector_store_path = "preliminary_vector_store"
    
    # Load or encode the vector store based on the presence of the FAISS index
    if os.path.exists(vector_store_path):
        embeddings = OpenAIEmbeddings()
        preliminary_inspection_vectorstore = FAISS.load_local(
            vector_store_path, embeddings, allow_dangerous_deserialization=True
        )
    else:
        preliminary_inspection_vectorstore = encode_preliminary_inspection_chapter(preliminary_inspection_chapter)
        preliminary_inspection_vectorstore.save_local(vector_store_path)

    # Set up a retriever to fetch the most relevant chunk
    preliminary_query_retriever = preliminary_inspection_vectorstore.as_retriever(search_kwargs={"k": 1})
    search_results = preliminary_query_retriever.invoke(query)
    if search_results:
        return search_results[0].page_content  
    else:
        return "No relevant content found."

@tool
def restoration_chapter_lookup(query: str) -> str:
    """
    Looks up relevant content in a pre-chunked book stored in FAISS based on the given query.
    
    Args:
        query (str): The query to search for in the book chunks.
        
    Returns:
        str: The most relevant content found in the book based on the query.
    """
    vector_store_path = "restoration_vector_store"
    
    # Load or encode the vector store based on the presence of the FAISS index
    if os.path.exists(vector_store_path):
        embeddings = OpenAIEmbeddings()
        restoration_chapter_vectorstore = FAISS.load_local(
            vector_store_path, embeddings, allow_dangerous_deserialization=True
        )
    else:
        restoration_chapter_vectorstore = encode_restoration_chapter(restoration_chapter)
        restoration_chapter_vectorstore.save_local(vector_store_path)

    # Set up a retriever to fetch the most relevant chunk
    restoration_query_retriever = restoration_chapter_vectorstore.as_retriever(search_kwargs={"k": 1})
    search_results = restoration_query_retriever.invoke(query)

    # Return the most relevant result if available
    if search_results:
        return search_results[0].page_content  
    else:
        return "No relevant content found."

