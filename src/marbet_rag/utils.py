import os

def format_docs_for_display(docs):
    """Helper function to format source document info for display."""
    if not docs:
        return "  (No sources retrieved)"
    sources_list = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'Unknown Source')
        page = doc.metadata.get('page', 'N/A')
        sources_list.append(f"  Source {i+1}: {source} (Page: {page})")
    return '\n'.join(sources_list) 