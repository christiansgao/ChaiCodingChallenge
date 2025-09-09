
# =========================
# PSEUDO-PYTHON: Chat + Image
# =========================

user_input = "Can you send me a selfie?"

# =========================
# 1. Full Prompt Injection
# =========================
persona_text = """
You are FunBot, a friendly chatbot.
- Always respond in a cheerful, humorous tone.
- You can describe yourself in text and generate images when requested.
- If the user asks for a selfie or scene, generate a suitable image using the AI image model.
"""

full_prompt = persona_text + "\nUser: " + user_input + "\nBot:"

# =========================
# 2. Generate Text Response
# =========================
response_text = llm.generate(prompt=full_prompt)
print("Bot Text Response:", response_text)

# =========================
# 3. Check if user requested an image
# =========================
def needs_image(user_input, response_text):
    triggers = ["selfie", "picture", "photo", "draw me", "show me"]
    return any(word in user_input.lower() for word in triggers)

if needs_image(user_input, response_text):
    # Create a prompt for the image model
    # Here we can optionally use context from the LLM response
    image_prompt = f"Generate a fun, cartoon-style selfie of FunBot. Context: {response_text}"

    # Generate image using an AI image model
    generated_image = image_model.generate(prompt=image_prompt)

    # Return both text and image to user
    print("Bot Image Response:", generated_image)
else:
    print("No image generated for this input.")