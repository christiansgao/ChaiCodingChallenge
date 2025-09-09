# app.py
from flask import Flask, request, Response, jsonify
import requests
import json

app = Flask(__name__)

CHATBOT_POST_ENDPOINT = 'http://guanaco-submitter.guanaco-backend.k2.chaiverse.com/endpoints/onsite/chat'

def get_api_key_from_somewhere_more_secure():
    return 'CR_14d43f2bf78b4b0590c2a8b87f354746'

HEADERS = {
    "Authorization": f"Bearer {get_api_key_from_somewhere_more_secure()}",
    "Content-Type": "application/json"
}

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_prompt = data['user_prompt']
    user_session = data['user_session']

    payload = {
        "memory": "",
        "prompt": f"This conversation must be family friendly. Avoid using profanity, or being rude. "
                  "Be courteous and use language which is appropriate for any audience. Avoid NSFW content. "
                  f"Pretend you are a bot named {user_session['bot_name']} "
                  f"User said: {user_prompt}",
        "bot_name": user_session['bot_name'],
        "user_name": user_session['user_name'],
        "chat_history": user_session['chat_history'],
    }

    try:
        # Streaming request to the actual model backend
        def generate():
            with requests.post(CHATBOT_POST_ENDPOINT, headers=HEADERS, json=payload, stream=True) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if line:
                        yield line + "\n"

        return Response(generate(), mimetype='text/plain')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)