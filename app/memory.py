from langchain_community.chat_message_histories import ChatMessageHistory

class Memory:
    def __init__(self):
        self.store = {}

    def get(self, session_id):
        """Retrieve session data by session_id."""
        return self.store.get(session_id)

    def set(self, session_id, value):
        """Store session data by session_id."""
        self.store[session_id] = value

    def get_session_history(self, session_id):
        """Retrieve or initialize chat history for a session."""
        # print("SessionID: ", session_id)
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()

        # print("Session History: ", self.store[session_id], type(self.store[session_id]))

        return self.store[session_id]

    def clear(self):
        """Clear the memory."""
        self.store = {}