import { useState, useEffect } from 'react'
import axios from 'axios'
import ChatWindow from './components/ChatWindow'
import InputBar from './components/InputBar'
import './App.css'

// Define the API endpoint URL
const API_URL = 'http://localhost:5000/api/chat' // Use the Flask API URL

// Regex to find and remove <think> blocks
const thinkRegex = /<think>[\s\S]*?<\/think>/gi; // Use global flag 'g'

function App() {
  const [messages, setMessages] = useState([])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)

  // Add initial welcome message with timestamp
  useEffect(() => {
    setMessages([
      {
        sender: 'ai',
        text: 'Welcome! Ask me anything about the event. 😊',
        timestamp: new Date().toISOString(),
        has_citations: false, // Default for initial message
        retrieved_context: [] // Ensure it has context array
      }
    ])
  }, [])

  const handleSendMessage = async (messageText) => {
    if (!messageText.trim() || isLoading) return

    const timestamp = new Date().toISOString() // Generate timestamp
    const userMessage = { sender: 'user', text: messageText, timestamp } // Add to user message
    // Keep previous messages and add the new user message
    const currentMessages = [...messages, userMessage]
    setMessages(currentMessages)
    setInputValue('')
    setIsLoading(true)
    setError(null) // Clear previous errors

    // --- Backend API Call ---
    try {
      // Prepare history for the API (send recent messages)
      // The API will apply its own window size limit
      const historyForApi = currentMessages.slice(0, -1).map(msg => ({
        // Only send sender and text to API, not timestamp or context
        sender: msg.sender,
        text: msg.text
      }))

      console.log("Sending to API:", { message: messageText, history: historyForApi })

      const response = await axios.post(API_URL, {
        message: messageText,
        history: historyForApi // Send the message history
      })

      console.log("API Response:", response.data)

      // Clean the raw answer from the API
      const rawAnswer = response.data.answer || "No answer received from API.";
      const cleanedAnswer = rawAnswer.replace(thinkRegex, '').trim();

      const aiMessage = {
        sender: 'ai',
        text: cleanedAnswer, // Use the cleaned answer text
        retrieved_context: response.data.retrieved_context || [],
        timestamp: new Date().toISOString(), // Add timestamp to AI response
        has_citations: response.data.has_citations || false // Store the flag
      }

      setMessages(prevMessages => [...prevMessages, aiMessage])

    } catch (err) {
      console.error("Error calling API:", err)
      let errorText = "Failed to get response from the assistant."
      if (err.response) {
        // Server responded with a status code outside 2xx range
        errorText += ` (Server error: ${err.response.status} ${err.response.data?.error || ''}`
      } else if (err.request) {
        // Request was made but no response received
        errorText += " (Server did not respond. Is the backend running?)"
      } else {
        // Something else happened in setting up the request
        errorText += ` (Error: ${err.message})`
      }
      setError(errorText) // Display error to user
      // Add an error message to the chat window
      const errorTimestamp = new Date().toISOString();
      const errorMessage = {
        sender: 'ai',
        text: `Error: ${errorText}`,
        timestamp: errorTimestamp,
        has_citations: false, // Errors don't have citations
        retrieved_context: [] // No context for errors
      }
      setMessages(prevMessages => [...prevMessages, errorMessage])
    } finally {
      setIsLoading(false)
    }
    // --- End API Call ---
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Marbet AI</h1>
      </header>
      <main className="chat-area">
        {/* Display error message if any */}
        {error && <div className="error-message">{error}</div>}
        <ChatWindow messages={messages} isLoading={isLoading} />
        <InputBar
          inputValue={inputValue}
          onInputChange={setInputValue}
          onSendMessage={handleSendMessage}
          isLoading={isLoading}
        />
      </main>
    </div>
  )
}

export default App
