from collections import deque
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

class ChatbotHistory:
    """Handles the conversation history between the chatbot and the user."""
    def __init__(self, max_size=5):
        self.buffer = deque(maxlen=max_size)

    def add_interaction(self, user_message: str, bot_response: str):
        """Stores user and bot interactions."""
        self.buffer.append(HumanMessage(content=user_message))
        self.buffer.append(AIMessage(content=bot_response))

    def get_history(self):
        """Returns the conversation history as a list."""
        return list(self.buffer)

    def get_history_as_string(self) -> str:
        """Returns the conversation history as a formatted string."""
        return "\n".join(
            [f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"Bot: {msg.content}"
            for msg in self.buffer]
        )

    def clear_history(self):
        """Clears the conversation history."""
        self.buffer.clear()


class ChatSessionManager:
    """Manages chat sessions and histories."""
    def __init__(self):
        self.store = {}

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Retrieves or initializes a chat history for a session."""
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]
