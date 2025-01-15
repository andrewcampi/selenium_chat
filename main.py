import streamlit as st
import openai
import os
import numpy as np
import pickle
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI clients
openai_client = openai.OpenAI()
groq_client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

def get_embeddings(texts, client=openai_client):
    """Get embeddings using OpenAI's text-embedding-3-small model"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return np.array([embedding.embedding for embedding in response.data])

def cosine_similarity(A, B):
    """Calculate cosine similarity between two sets of vectors"""
    A_norm = A / np.linalg.norm(A, axis=1)[:, np.newaxis]
    B_norm = B / np.linalg.norm(B, axis=1)[:, np.newaxis]
    return np.dot(A_norm, B_norm.T)

class SeleniumAssistant:
    def __init__(self, docs_path="selenium_docs.txt"):
        self.embeddings_file = "selenium_embeddings.pkl"
        self.chunks_file = "selenium_chunks.pkl"
        
        # Load or create embeddings and chunks
        if Path(self.embeddings_file).exists() and Path(self.chunks_file).exists():
            self.load_cached_data()
        else:
            self.chunks = self.process_text_file(docs_path)
            self.embeddings = self.create_embeddings(self.chunks)
            self.save_cached_data()

    def process_text_file(self, file_path, chunk_size=1000):
        """Process text file into chunks for embedding."""
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Split text into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Combine paragraphs into chunks
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            paragraph_length = len(paragraph)
            
            if current_length + paragraph_length > chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [paragraph]
                current_length = paragraph_length
            else:
                current_chunk.append(paragraph)
                current_length += paragraph_length
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks

    def create_embeddings(self, chunks):
        """Generate embeddings using OpenAI's API."""
        return get_embeddings(chunks)

    def save_cached_data(self):
        """Save embeddings and chunks to disk."""
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(self.embeddings, f)
        with open(self.chunks_file, 'wb') as f:
            pickle.dump(self.chunks, f)

    def load_cached_data(self):
        """Load embeddings and chunks from disk."""
        with open(self.embeddings_file, 'rb') as f:
            self.embeddings = pickle.load(f)
        with open(self.chunks_file, 'rb') as f:
            self.chunks = pickle.load(f)

    def get_relevant_context(self, query, num_chunks=3):
        """Get most relevant chunks of text for a given query."""
        query_embedding = get_embeddings([query])[0]
        query_embedding = query_embedding.reshape(1, -1)

        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # Get top chunks
        top_indices = np.argsort(similarities)[-num_chunks:][::-1]
        context = "\n\n".join([self.chunks[i] for i in top_indices])
        
        return context

    def get_response(self, query, context):
        """Get response from LLM using context with failover between Groq and OpenAI."""
        system_prompt = """You are an expert Selenium automation engineer. Your task is to:
1. Answer questions about Selenium with accurate, working code examples
2. Translate English descriptions into working Selenium Python code
3. Provide clear explanations of Selenium concepts

Always format code examples using Python markdown syntax. Include complete imports and setup when relevant.
Base your answers on the provided documentation context, and if something isn't covered in the context,
use standard Selenium best practices."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context from Selenium documentation:\n{context}\n\nUser question: {query}"}
        ]

        # Try Groq first
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.2,
                max_tokens=1500
            )
            return response.choices[0].message.content
            
        except Exception as e:
            st.warning("Falling back to OpenAI due to Groq error...")
            # Fallback to OpenAI
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.2,
                    max_tokens=1500
                )
                return response.choices[0].message.content
                
            except Exception as e:
                return f"Error: Both Groq and OpenAI failed to generate a response. Please try again later. Error: {str(e)}"

def main():
    st.title("🔍 Selenium Code Assistant")
    st.write("Ask questions about Selenium or describe what you want to automate!")

    # Initialize SeleniumAssistant (only once)
    if 'assistant' not in st.session_state:
        with st.spinner("Initializing Selenium Assistant..."):
            st.session_state.assistant = SeleniumAssistant()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Describe what you want to automate..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Get relevant context
                context = st.session_state.assistant.get_relevant_context(prompt)
                # Generate response
                response = st.session_state.assistant.get_response(prompt, context)
                st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()