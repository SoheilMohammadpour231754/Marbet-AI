import React from 'react';
import './InputBar.css'; // We'll create this CSS file next

function InputBar({ inputValue, onInputChange, onSendMessage, isLoading }) {

  // Handle Enter key press in the textarea
  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault(); // Prevent newline in textarea
      handleSendClick();
    }
  };

  // Handle button click
  const handleSendClick = () => {
    onSendMessage(inputValue);
  }

  return (
    <div className="input-bar">
      <textarea
        value={inputValue}
        onChange={(e) => onInputChange(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={isLoading ? "Waiting for response..." : "Type your message here... (Shift+Enter for newline)"}
        rows="2" // Start with 2 rows, can expand
        disabled={isLoading} // Disable input while AI is thinking
      />
      <button onClick={handleSendClick} disabled={isLoading || !inputValue.trim()}>
        {isLoading ? 'Sending...' : 'Send'}
      </button>
    </div>
  );
}

export default InputBar; 