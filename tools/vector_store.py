"""
Vector store for dynamic few-shot example retrieval.
"""

from typing import List, Dict
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from loguru import logger
from config import settings
import os


class FewShotRetriever:
    """
    Manages a vector store of SQL examples for dynamic few-shot learning.
    """
    
    def __init__(self):
        self.enabled = settings.enable_dynamic_few_shot
        
        if not self.enabled:
            logger.info("Dynamic few-shot learning disabled")
            return
        
        # Initialize embeddings with HuggingFace model (local)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model
        )
        
        # Initialize vector store
        persist_directory = settings.vector_store_path
        os.makedirs(persist_directory, exist_ok=True)
        
        self.vectorstore = Chroma(
            collection_name=settings.chroma_collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        
        logger.info(f"Few-shot retriever initialized with ChromaDB")
    
    def add_example(self, question: str, sql: str, explanation: str = None, 
                    schema_context: str = None, complexity: str = "medium"):
        """
        Add a new SQL example to the vector store.
        
        Args:
            question: Natural language question
            sql: Corresponding SQL query
            explanation: Optional explanation
            schema_context: Optional schema info
            complexity: Difficulty level (simple, medium, complex)
        """
        if not self.enabled:
            return
        
        try:
            doc = Document(
                page_content=question,
                metadata={
                    "sql": sql,
                    "explanation": explanation or "",
                    "schema_context": schema_context or "",
                    "complexity": complexity
                }
            )
            
            self.vectorstore.add_documents([doc])
            logger.info(f"Added example: {question[:50]}...")
            
        except Exception as e:
            logger.error(f"Error adding example: {e}")
    
    def add_examples_batch(self, examples: List[Dict]):
        """
        Add multiple examples at once.
        
        Args:
            examples: List of example dicts with 'question', 'sql', etc.
        """
        if not self.enabled:
            return
        
        try:
            docs = []
            for ex in examples:
                doc = Document(
                    page_content=ex["question"],
                    metadata={
                        "sql": ex.get("sql", ""),
                        "explanation": ex.get("explanation", ""),
                        "schema_context": ex.get("schema_context", ""),
                        "complexity": ex.get("complexity", "medium")
                    }
                )
                docs.append(doc)
            
            self.vectorstore.add_documents(docs)
            logger.info(f"Added {len(docs)} examples to vector store")
            
        except Exception as e:
            logger.error(f"Error adding batch examples: {e}")
    
    def retrieve(self, question: str, k: int = None) -> List[Dict]:
        """
        Retrieve most relevant examples for a question.
        
        Args:
            question: User's question
            k: Number of examples to retrieve
            
        Returns:
            List of example dicts
        """
        if not self.enabled:
            return []
        
        k = k or settings.few_shot_examples_count
        
        try:
            # Similarity search
            results = self.vectorstore.similarity_search(question, k=k)
            
            examples = []
            for doc in results:
                examples.append({
                    "question": doc.page_content,
                    "sql": doc.metadata.get("sql", ""),
                    "explanation": doc.metadata.get("explanation", ""),
                    "schema_context": doc.metadata.get("schema_context", ""),
                    "complexity": doc.metadata.get("complexity", "medium")
                })
            
            logger.info(f"Retrieved {len(examples)} similar examples")
            return examples
            
        except Exception as e:
            logger.error(f"Error retrieving examples: {e}")
            return []
    
    def clear(self):
        """Clear all examples from vector store."""
        if self.enabled:
            # Delete and recreate collection
            self.vectorstore.delete_collection()
            logger.info("Vector store cleared")


# Global retriever instance
few_shot_retriever = FewShotRetriever()


def seed_examples():
    """
    Seed the vector store with common SQL patterns.
    This should be called during setup with your domain-specific examples.
    """
    if not settings.enable_dynamic_few_shot:
        return
    
    default_examples = [
        {
            "question": "What is the total revenue for each product category?",
            "sql": """SELECT 
    category,
    SUM(price * quantity) as total_revenue
FROM products p
JOIN sales s ON p.product_id = s.product_id
GROUP BY category
ORDER BY total_revenue DESC""",
            "explanation": "Join products with sales and aggregate by category",
            "complexity": "simple"
        },
        {
            "question": "Find customers who made purchases in the last 30 days but not in the previous 30 days",
            "sql": """SELECT DISTINCT c.customer_id, c.name
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_date >= CURRENT_DATE - INTERVAL '30 days'
  AND c.customer_id NOT IN (
    SELECT customer_id 
    FROM orders 
    WHERE order_date >= CURRENT_DATE - INTERVAL '60 days'
      AND order_date < CURRENT_DATE - INTERVAL '30 days'
  )""",
            "explanation": "Use subquery to exclude customers from previous period",
            "complexity": "complex"
        },
        {
            "question": "Calculate the 3-month rolling average of sales",
            "sql": """SELECT 
    DATE_TRUNC('month', sale_date) as month,
    AVG(sale_amount) OVER (
        ORDER BY DATE_TRUNC('month', sale_date)
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) as rolling_3month_avg
FROM sales
ORDER BY month""",
            "explanation": "Use window function with ROWS BETWEEN for rolling average",
            "complexity": "complex"
        }
    ]
    
    few_shot_retriever.add_examples_batch(default_examples)
    logger.info(f"Seeded {len(default_examples)} default examples")
