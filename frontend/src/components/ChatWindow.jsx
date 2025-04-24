import React, { useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import './ChatWindow.css'; // We'll create this CSS file next

// Helper function to extract filename from path
function getFilename(path) {
  if (!path) return 'Unknown Source';
  return path.split('\\').pop().split('/').pop(); // Handles both Windows and Unix paths
}

// Helper function to format timestamp (e.g., HH:MM)
function formatTime(isoString) {
  if (!isoString) return '';
  try {
    const date = new Date(isoString);
    return date.toLocaleTimeString(navigator.language, { hour: '2-digit', minute:'2-digit', hour12: false });
  } catch (e) {
    console.error("Error formatting time:", e);
    return ''; // Return empty string on error
  }
}

function ChatWindow({ messages, isLoading }) {
  const messagesEndRef = useRef(null);

  // Function to scroll to the bottom of the chat window
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }

  // Scroll to bottom whenever messages change
  useEffect(() => {
    // Slightly delay scroll to allow layout adjustments
    const timer = setTimeout(scrollToBottom, 100);
    return () => clearTimeout(timer);
  }, [messages]);

  // Add Typing Indicator Component (inline for simplicity)
  const TypingIndicator = () => (
    <div className="message ai">
      <div className="message-bubble typing-indicator">
        <span></span>
        <span></span>
        <span></span>
      </div>
    </div>
  );

  return (
    <div className="chat-window">
      {messages.map((message, index) => {
        const visibleAnswer = message.text;

        return (
          <div key={index} className={`message ${message.sender}`}>
            <div className="message-bubble">
              {/* Render visible answer using ReactMarkdown */}
              <div className="markdown-content">
                <ReactMarkdown>{visibleAnswer}</ReactMarkdown>
              </div>

              {/* Collapsible Sources - Render only if citations were found */}
              {/* {message.sender === 'ai' && message.has_citations && message.retrieved_context && message.retrieved_context.length > 0 && (
                <details className="sources-details">
                  <summary className="sources-summary">Show Sources Considered ({message.retrieved_context.length})</summary>
                  <div className="sources-container">
                    <ul className="sources-list">
                      {message.retrieved_context.map((retrievedDoc, srcIndex) => {
                        const filename = getFilename(retrievedDoc.metadata?.source);
                        const page = retrievedDoc.metadata?.page;
                        return (
                          <li key={srcIndex} className="source-item">
                            <a href="#" onClick={(e) => e.preventDefault()} className="source-link">
                              {filename}
                            </a>
                            {page !== undefined && ` (Page: ${page + 1})`}
                          </li>
                        );
                      })}
                    </ul>
                  </div>
                </details>
              )} */}

              {/* Timestamp */}
              {message.timestamp && (
                <span className="message-timestamp">{formatTime(message.timestamp)}</span>
              )}
            </div>
          </div>
        );
      })}

      {/* Conditionally render Typing Indicator */}
      {isLoading && <TypingIndicator />}

      <div ref={messagesEndRef} />
    </div>
  );
}

export default ChatWindow; 