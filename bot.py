from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

model_name = "EleutherAI/gpt-neo-125M"  # Puedes usar "gpt2" si prefieres
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)

def chat():
    print("¡Hola! Soy un chatbot. Escribe 'salir' para terminar la conversación.")
    while True:
        user_input = input("Tú: ")
        if user_input.lower() == "salir":
            print("Chatbot: ¡Adiós! Fue un placer ayudarte.")
            break

        response = chatbot(user_input, max_length=50, num_return_sequences=1)
        print(f"Chatbot: {response[0]['generated_text']}")

if __name__ == "__main__":
    chat()