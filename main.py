# main.py
from response_generator import ResponseGenerator

def main():
    bot = ResponseGenerator()
    print("Airline Support Bot (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = bot.generate(user_input)
        print(f"Bot: {response}\n")

if __name__ == "__main__":
    main()
