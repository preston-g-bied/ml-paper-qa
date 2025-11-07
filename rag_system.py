"""
ML Research Paper QA System
A simple RAG system for querying machine learning research papers.
"""

import os
from typing import List, Dict
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# load environment variables
load_dotenv()

class PaperQASystem:
    """Simple RAG system for ML research papers."""

    def __init__(self, model_name: str = "gpt-3.5-turbo", chunk_size: int= 1000, chunk_overlap: int = 200):
        """
        Initialize the QA system.
        
        Args:
            model_name: OpenAI model to use (gpt-3.5-turbo is cheap)
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks for context
        """
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore = None
        self.qa_chain = None

        # initialize embeddings (used for vectorizing text)
        self.embeddings = OpenAIEmbeddings()

        # initialize LLM (using temperature=0 for consistent, factual responses)
        self.llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=0
        )

    def load_papers(self, pdf_paths: List[str]) -> int:
        """
        Load and process PDF papers into the vector store.
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Returns:
            Number of chunks created
        """
        print(f"Loading {len(pdf_paths)} papers...")

        # load all documents
        documents = []
        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                print(f"Warning: {pdf_path} not found, skipping...")
                continue

            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            # add filename to metadata for citation
            for doc in docs:
                doc.metadata['source'] = Path(pdf_path).name

            documents.extend(docs)
            print(f"    Loaded {Path(pdf_path).name}: {len(docs)} pages")

        if not documents:
            raise ValueError("No documents were loaded successfully")
        
        # split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} text chunks")

        # create vector store
        print("Creating embeddings and vector store...")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)

        # initialize QA chain
        self._setup_qa_chain()

        print("Papers loaded successfully!")
        return len(chunks)
    
    def _setup_qa_chain(self):
        """Set up the question-answering chain with custom prompt."""

        # custom prompt to encourage citing sources
        prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Always cite which paper(s) you're referencing in your answer.

Context:
{context}

Question: {question}

Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # create retrieval QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",     # puts all retrieved docs into context
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}  # retrieve top 3 most relevant chunks
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

    def ask(self, question: str) -> Dict:
        """
        Ask a question about the loaded papers.
        
        Args:
            question: Natural language question
            
        Returns:
            Dictionary with 'answer' and 'sources'
        """
        if self.qa_chain is None:
            raise ValueError("No papers loaded. Call load_papers() first.")
        
        # get answer from QA chain
        result = self.qa_chain.invoke({"query": question})

        # format sources
        sources = []
        for doc in result['source_documents']:
            source_info = {
                'content': doc.page_content[:200] + "...",  # first 200 chars
                'paper': doc.metadata.get('source', 'Unknown'),
                'page': doc.metadata.get('page', 'Unknown')
            }
            sources.append(source_info)

        return {
            'answer': result['result'],
            'sources': sources
        }
    
    def save_vectorstore(self, path: str = "vectorstore"):
        """Save the vector store to disk."""
        if self.vectorstore is None:
            raise ValueError("No vector store to save")
        self.vectorstore.save_local(path)
        print(f"Vector store saved to {path}")

    def load_vectorstore(self, path: str = "vectorstore"):
         """Load a previously saved vector store."""
         self.vectorstore = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
         self._setup_qa_chain()
         print(f"Vector store loaded from {path}")

def main():
    """Example usage of the PaperQASystem."""

    # check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found!")
        return
    
    # initialize system
    qa_system = PaperQASystem()

    # example: load papers from data dictionary
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    pdf_files = list(data_dir.glob("*.pdf"))

    if not pdf_files:
        print("\nNo PDF files found in the 'data' directory.")
        return
    
    # load papers
    qa_system.load_papers([str(f) for f in pdf_files])

    # interactive Q&A loop
    print("\n" + "="*60)
    print("Ready! Ask questions about your papers (type 'quit' to exit)")
    print("="*60 + "\n")

    while True:
        question = input("\nQuestion: ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            break

        if not question:
            continue

        try:
            result = qa_system.ask(question)

            print(f"\nAnswer: {result['answer']}")
            print("\nSources:")
            for i, source in enumerate(result['sources'], 1):
                print(f"  [{i}] {source['paper']} (page {source['page']})")
                print(f"      {source['content']}\n")

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
