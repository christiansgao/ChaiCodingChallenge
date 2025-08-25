"""
Assumptions:
1. If CHAI engine uses history very well, EX: if the bot explains its past history and personality within the chat history,
all this info does not necessarily need to be re-injected into the prompt every API call.

Limitations:
1. This bot barely just "works" its got no features or efficiency consideration.
2. Bot and chat history does not get saved and dies when the session ends
3. Response does not come as a stream 1 token at a time, it comes all at once as a request response due to limitations of simple python libs
4. No GUI or images at all!
"""
import requests

CHATBOT_POST_ENDPOINT = 'http://guanaco-submitter.guanaco-backend.k2.chaiverse.com/endpoints/onsite/chat'


class UserChatSession:
    def __init__(
            self,
            bot_name: str,
            bot_description: str,
            background_story: str,
            user_name: str,
            chat_history: list[dict]
    ) -> None:
        self.bot_name = bot_name
        self.bot_description = bot_description
        self.background_story = background_story
        self.user_name = user_name
        self.chat_history = chat_history


def get_api_key_from_somewhere_more_secure():
    # In real life, store code-store key in env variable or config or fast access db
    return 'CR_14d43f2bf78b4b0590c2a8b87f354746'


HEADERS = {
    "Authorization": f"Bearer {get_api_key_from_somewhere_more_secure()}",
    "Content-Type": "application/json"
}


def chatbot_response(user_chat_session: UserChatSession, user_prompt: str):
    payload = {
        "memory": "",
        "prompt": f"This conversation must be family friendly. Avoid using profanity, or being rude. " 
                  "Be courteous and use language which is appropriate for any audience. Avoid NSFW content. " 
                  f"Pretend you are a bot named {user_chat_session.bot_name} "
                  f"User said: {user_prompt}",
        "bot_name": user_chat_session.bot_name,
        "user_name": user_chat_session.user_name,
        "chat_history": user_chat_session.chat_history,
    }

    try:
        response = requests.post(CHATBOT_POST_ENDPOINT, headers=HEADERS, json=payload)
        response.raise_for_status()
        bot_reply = response.json().get("message", f"{user_chat_session.bot_name} Refuses to Reply")
        print(f"{user_chat_session.bot_name}: {bot_reply}")
        user_chat_session.chat_history.append({"sender": user_chat_session.user_name, "message": user_prompt})
        user_chat_session.chat_history.append({"sender": user_chat_session.bot_name, "message": bot_reply})

    except Exception as e:
        print(f"[Error]: {e}")


def get_bot_background_story(bot_name, bot_description):
    payload = {
        "memory": "",
        "prompt": f"This conversation must be family friendly. Avoid using profanity, or being rude. "
                  f"Be courteous and use language which is appropriate for any audience. Avoid NSFW content. "
                  f"User is creating a bot named: {bot_name}, with description: {bot_description}, generate a rich background story based on that description",
        "bot_name": bot_name,
        "user_name": "default",
        "chat_history": [],
    }

    try:
        response = requests.post(CHATBOT_POST_ENDPOINT, headers=HEADERS, json=payload)
        response.raise_for_status()
        bot_background_story = response.json().get("message", "ERROR")
        if bot_background_story == "ERROR":
            raise Exception("Unable to generate a rich background story.")
        return bot_background_story

    except Exception as e:
        print(f"[Error]: {e}")


def main():
    user_name = input("Select User Name: ").strip()
    bot_name = input("Give a name for your Bot: ").strip()
    bot_description = input("Give a Description for your Bot: ").strip()

    bot_background_story = get_bot_background_story(bot_name, bot_description)

    bot_first_message = f"Hi I am: {bot_name}, you created me here is my story: {bot_background_story}, you may have created a monster, but whats up?"
    chat_history = [{"sender": bot_name, "message": bot_first_message}]

    usr_session = UserChatSession(bot_name, bot_description, bot_background_story, user_name, chat_history)

    while True:
        user_prompt = input(f"{user_name}: ").strip()
        if user_prompt.lower() == "exit":
            break

        chatbot_response(usr_session, user_prompt)


if __name__ == "__main__":
    main()
