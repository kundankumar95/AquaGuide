document.addEventListener('DOMContentLoaded', () => {
    const sendButton = document.getElementById('sendButton');
    const voiceButton = document.getElementById('voiceButton');
    const userInput = document.getElementById('userInput');
    const responseArea = document.getElementById('responseArea');

    // Handle text input and send button click
    sendButton.addEventListener('click', () => {
        const query = userInput.value.trim();
        if (query) {
            handleUserQuery(query);
            userInput.value = '';
        }
    });

    // Handle voice button click
    voiceButton.addEventListener('click', () => {
        const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
        recognition.lang = 'en-US';
        recognition.start();

        recognition.onresult = (event) => {
            const query = event.results[0][0].transcript;
            handleUserQuery(query);
        };

        recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
        };
    });

    // Function to handle user queries
    async function handleUserQuery(query) {
        responseArea.innerHTML += `<p><strong>You:</strong> ${query}</p>`;

        try {
            const response = await fetch(`http://localhost:5000/classify?query=${encodeURIComponent(query)}`);
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            const data = await response.json();
            const assistantResponse = data.response;
            const probability = data.probability;

            // Display the assistant's response
            responseArea.innerHTML += `<p><strong>Assistant:</strong> ${assistantResponse} (Probability: ${probability})</p>`;
        } catch (error) {
            console.error('There was a problem with the fetch operation:', error);
            responseArea.innerHTML += `<p><strong>Assistant:</strong> Sorry, there was an error processing your request.</p>`;
        }

        responseArea.scrollTop = responseArea.scrollHeight; // Scroll to bottom
    }
});

