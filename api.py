#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Flask API for the Marbet RAG Chatbot."""

import traceback
import re # Import regex module
import os # Import os module for path manipulation
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_core.messages import AIMessage, HumanMessage

# Import the setup function and potentially config if needed later
from src.marbet_rag.retrieval import setup_chatbot
from src.marbet_rag.utils import format_docs_for_display # For potential source formatting
import config # Import config to access HISTORY_WINDOW_SIZE

# --- Initialize Flask App and CORS ---
app = Flask(__name__)
# Allow requests from the React frontend development server (default port 5173)
# In production, restrict this to the actual frontend domain
CORS(app, resources={r"/api/*": {"origins": "*"}}) # Allow all origins for now

# --- Initialize Chatbot --- #
# This runs once when the Flask app starts
try:
    print("Starting chatbot initialization for API...")
    # Initialize the RAG chain globally
    rag_chain = setup_chatbot()
    print("Chatbot initialization complete. API is ready.")
except Exception as e:
    print(f"FATAL ERROR: Could not initialize chatbot for API: {e}")
    rag_chain = None # Ensure rag_chain is None if setup fails

# Regex to find citations like [Source: file.pdf, Page: 1] or [Source: file.pdf]
CITATION_REGEX = re.compile(r"\[Source:\s*(.*?)(?:,\s*Page:\s*\d+)?\s*\]", re.IGNORECASE)

# Helper function to get base filename
def get_base_filename(path):
    if not path: return None
    return os.path.basename(path)

# --- API Endpoint --- #
@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    """Handles chat requests from the frontend."""
    if rag_chain is None:
        return jsonify({"error": "Chatbot is not initialized. Please check server logs."}), 500

    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Missing 'message' in request body"}), 400

        user_query = data['message']
        # History from frontend should be a list of objects like {sender: 'user'/'ai', text: '...'}
        history_data = data.get('history', [])

        # Prepare history for the LangChain model
        history_for_chain = []
        # Limit history based on config
        pairs_to_keep = config.HISTORY_WINDOW_SIZE // 2
        start_index = max(0, len(history_data) - pairs_to_keep * 2)
        temp_history = history_data[start_index:]

        for msg in temp_history:
            if msg.get('sender') == 'user':
                history_for_chain.append(HumanMessage(content=msg.get('text', '')))
            elif msg.get('sender') == 'ai':
                history_for_chain.append(AIMessage(content=msg.get('text', '')))

        print(f"API received query: '{user_query}' with {len(history_for_chain)} history messages.")

        # Invoke the RAG chain
        response = rag_chain.invoke({
            "input": user_query,
            "chat_history": history_for_chain
        })

        ai_answer = response.get("answer", "Sorry, I encountered an issue generating an answer.")
        retrieved_docs = response.get('context', [])

        # --- Debugging: Log Retrieved Document Content ---
        print("\n--- Retrieved Context for LLM ---")
        if retrieved_docs:
            for i, doc in enumerate(retrieved_docs):
                source_name = doc.metadata.get('source', 'Unknown')
                page_num = doc.metadata.get('page', 'N/A')
                print(f"Doc {i+1}: Source='{source_name}', Page={page_num}")
                # Corrected f-string: Use a temporary variable for the replaced string
                content_snippet = doc.page_content[:150].replace('\n', ' ')
                print(f"   Content Snippet: {content_snippet}...")
        else:
            print("No documents retrieved.")
        print("--- End Retrieved Context ---\n")
        # --- End Debugging ---

        print(f"API generated answer: '{ai_answer[:100]}...'") # Log snippet

        # --- Filter sources based on citations in the answer --- #
        cited_filenames = set()
        matches = CITATION_REGEX.findall(ai_answer)
        for match in matches:
            raw_filename = match.strip()
            base_filename = get_base_filename(raw_filename)
            if base_filename:
                cited_filenames.add(base_filename)
        
        # Determine if citations were actually found
        has_citations = len(cited_filenames) > 0
        print(f"Citations found in answer: {has_citations}")
        
        cited_sources_data = []
        added_doc_identifiers = set()
        if retrieved_docs and cited_filenames:
            for doc in retrieved_docs:
                doc_source_path = doc.metadata.get('source')
                doc_base_filename = get_base_filename(doc_source_path)
                doc_identifier = f"{doc_source_path}_p{doc.metadata.get('page')}"
                if doc_base_filename in cited_filenames and doc_identifier not in added_doc_identifiers:
                    cited_sources_data.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    })
                    added_doc_identifiers.add(doc_identifier)

        print(f"Sending {len(cited_sources_data)} cited sources to frontend.")
        # --- End Filtering --- #

        # --- Prepare list of ALL retrieved documents for frontend --- #
        retrieved_context_for_frontend = []
        if retrieved_docs:
            for doc in retrieved_docs:
                 retrieved_context_for_frontend.append({
                    "metadata": doc.metadata
                 })
        print(f"Sending metadata for {len(retrieved_context_for_frontend)} originally retrieved documents to frontend.")
        # --- End preparing retrieved context --- #

        # --- Clean citations from the final answer text --- #
        cleaned_answer = CITATION_REGEX.sub('', ai_answer).strip()
        print(f"API generated cleaned answer: '{cleaned_answer[:100]}...'" )
        # --- End Cleaning --- #

        return jsonify({
            "answer": cleaned_answer,
            "retrieved_context": retrieved_context_for_frontend,
            "has_citations": has_citations
        })

    except Exception as e:
        print(f"Error processing API request: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred."}), 500

# --- Run the App --- #
if __name__ == '__main__':
    # Note: Use a production WSGI server like Gunicorn or Waitress for deployment
    print("Starting Flask development server...")
    app.run(host='0.0.0.0', port=5000, debug=False) # Run on port 5000, accessible externally 