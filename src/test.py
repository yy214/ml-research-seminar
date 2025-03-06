from ollama import chat


class LlamaChat:
    def get_response(self, messages):
        response = chat(
            model="llama3.2",
            messages=messages,
        )
        return response.message.content


print("Llama LLM Chat - Type 'exit' to quit.")


llamaChat = LlamaChat()
messages = []

while True:
    user_input = input("\nYou: ")

    if user_input.lower() in ["exit", "quit"]:
        print("\nExiting. Goodbye!")
        break

    messages.append({"role": "user", "content": user_input})
    response = llamaChat.get_response(messages)
    messages.append({"role": "assistant", "content": response})
    print(f"Llama: {response}")
