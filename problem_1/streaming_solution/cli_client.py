# cli_client.py
import requests

BACKEND_URL = "http://localhost:5000/chat"

class UserChatSession:
    def __init__(self, bot_name, bot_description, background_story, user_name):
        self.bot_name = bot_name
        self.bot_description = bot_description
        self.background_story = background_story
        self.user_name = user_name
        self.chat_history = []

def main():
    user_name = input("Select User Name: ").strip()
    bot_name = input("Give a name for your Bot: ").strip()
    bot_description = input("Give a Description for your Bot: ").strip()

    # Faking the background story for simplicity (or call real generation endpoint)
    background_story = f"{bot_name} is a bot designed to: {bot_description}"

    session = UserChatSession(bot_name, bot_description, background_story, user_name)
    bot_first_message = f"Hi I am: {bot_name}, you created me. Here is my story: {background_story}. What's up?"
    session.chat_history.append({"sender": bot_name, "message": bot_first_message})
    print(f"{bot_name}: {bot_first_message}")

    while True:
        user_prompt = input(f"{user_name}: ").strip()
        if user_prompt.lower() == "exit":
            break

        payload = {
            "user_prompt": user_prompt,
            "user_session": {
                "bot_name": session.bot_name,
                "bot_description": session.bot_description,
                "background_story": session.background_story,
                "user_name": session.user_name,
                "chat_history": session.chat_history
            }
        }

        try:
            print(f"{session.bot_name}: ", end='', flush=True)
            response = requests.post(BACKEND_URL, json=payload, stream=True)
            response.raise_for_status()
            bot_reply = ""
            for line in response.iter_lines(decode_unicode=True):
                print(line, end='', flush=True)
                bot_reply += line

            print()
            session.chat_history.append({"sender": session.user_name, "message": user_prompt})
            session.chat_history.append({"sender": session.bot_name, "message": bot_reply})

        except Exception as e:
            print(f"\n[Error]: {e}")

if __name__ == "__main__":
    main()
