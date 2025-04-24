import os
import traceback
import config
from langchain_core.messages import AIMessage, HumanMessage

# Import functions from the marbet_rag package
# Remove unused imports related to direct setup
# from src.marbet_rag.data_processing import load_documents, split_documents
# from src.marbet_rag.retrieval import get_vector_store, setup_rag_chain
from src.marbet_rag.retrieval import setup_chatbot # Import the new setup function
from src.marbet_rag.utils import format_docs_for_display


if __name__ == "__main__":
    # --- Setup Chatbot (Handles Vector Store and RAG Chain Init) ---
    # Initialization prints are now handled inside setup_chatbot
    try:
        rag_chain = setup_chatbot()
    except SystemExit: # Catch the exit(1) from setup_chatbot on failure
        print("Exiting due to setup failure.")
                exit(1)
    # except Exception as setup_error:
    #     print(f"Unhandled error during setup: {setup_error}")
    #     traceback.print_exc()
    #     exit(1)

    # --- Querying Loop (CLI) ---
    print("\n--- Marbet Event Assistant CLI Ready ---")
        print("Ask questions about the event (type 'quit' to exit).")

    chat_history = [] # Keep history as simple list of strings for CLI

        while True:
            user_query = input("\nYour question: ").strip()
            if user_query.lower() == 'quit':
                break
            if not user_query:
                continue

            print("Processing...")
            try:
            # Prepare history for the chain (convert string list to Message list)
                history_for_chain = []
                pairs_to_keep = config.HISTORY_WINDOW_SIZE // 2
                start_index = max(0, len(chat_history) - pairs_to_keep * 2)
                temp_history = chat_history[start_index:]
                for i in range(0, len(temp_history), 2):
                    if i + 1 < len(temp_history):
                        history_for_chain.append(HumanMessage(content=temp_history[i]))
                        history_for_chain.append(AIMessage(content=temp_history[i+1]))

                # Invoke the RAG chain
                response = rag_chain.invoke({
                    "input": user_query,
                    "chat_history": history_for_chain
                })

                # Access answer and context from the response dictionary
                ai_answer = response.get("answer", "Sorry, I could not generate an answer.")
                retrieved_docs = response.get('context', [])

                print("\nAssistant:")
                print(ai_answer)

                # Display source documents used
                print("\nRetrieved Sources:")
                print(format_docs_for_display(retrieved_docs))

            # Update chat history (simple string list)
                chat_history.append(user_query)
                chat_history.append(ai_answer)

            except Exception as e:
                print(f"\nAn error occurred during query processing: {e}")
                traceback.print_exc()

    print("\n--- Exiting CLI ---") 