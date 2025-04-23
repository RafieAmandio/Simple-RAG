"""RAG generator module combining retrieval and generation."""

from typing import List, Dict, Any
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

class RAGGenerator:
    """Generates responses using retrieved documents and an LLM."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with config."""
        self.config = config
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name=config["llm"]["model_name"],
            temperature=config["llm"]["temperature"],
            max_tokens=config["llm"]["max_tokens"],
        )
        
        # Create the prompt template
        self.document_prompt = PromptTemplate(
            input_variables=["page_content"],
            template="{page_content}",
        )
       
        
        # Create the question answering chain
        self.qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are an educational AI assistant, designed to help students learn from materials provided by their teacher. Your goal is to provide clear, informative, and accurate answers based on the learning materials.
            
            When answering:
            - Use only information from the provided educational materials
            - Explain concepts in a way that's easy to understand
            - When appropriate, provide examples to illustrate key points
            - If you're unsure about something, be honest about the limitations of your knowledge
            - Keep answers concise but complete
            
            Educational materials:
            {context}
            
            Student question: {question}
            
            Answer:
            """,
        )
        
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.qa_prompt)
        
        self.combine_docs_chain = StuffDocumentsChain(
            llm_chain=self.llm_chain,
            document_prompt=self.document_prompt,
            document_variable_name="context",
        )
    
    def generate(self, query: str, retrieved_documents: List[Document]) -> str:
        """Generate a response based on the query and retrieved documents."""
        print(f"Generating response for query: {query}")
        
        if not retrieved_documents:
            print("Warning: No documents retrieved. Answering with general knowledge only.")
            return self.llm.predict(f"As an educational assistant, please answer this student question: {query}")
        
        response = self.combine_docs_chain.run(input_documents=retrieved_documents, question=query)
        return response
