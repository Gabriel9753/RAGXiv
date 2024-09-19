function insertCommand() {
    const command = 'Hello, Streamlit!';
    const chatInput = parent.document.querySelector('.stChatInputTextArea');
    console.log(chatInput);
    console.log(command);
    if (chatInput) {
        chatInput.value = command;
        chatInput.dispatchEvent(new Event('input', { bubbles: true }));
        chatInput.focus();
    }
}