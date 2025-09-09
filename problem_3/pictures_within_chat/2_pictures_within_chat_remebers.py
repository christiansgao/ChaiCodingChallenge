# =========================
# PSEUDO-PYTHON: Multi-Turn Chat + Consistent Images
# =========================

# Initialize conversation memory
conversation_memory = []

# Initialize bot identity (appearance info)
bot_identity_description = "FunBot is a friendly, cartoon-style robot with a blue hat and big smile."

# Example chat loop
for user_input in [
    "Hi FunBot!",
    "Can you send me a selfie?",
    "Now draw yourself waving!"
]:

    # =========================
    # 1. Build full prompt with memory
    # =========================
    persona_text = f"""
    You are FunBot, a friendly chatbot.
    - Always respond cheerfully and humorously.
    - You can describe yourself in text and generate images when requested.
    - Your appearance: {bot_identity_description}
    - Remember past conversation to keep continuity.
    """

    # Include last N turns from memory (if any)
    memory_context = "\n".join(conversation_memory[-5:])  # last 5 turns
    full_prompt = persona_text + "\nConversation History:\n" + memory_context + "\nUser: " + user_input + "\nBot:"

    # =========================
    # 2. Generate text response
    # =========================
    response_text = llm.generate(prompt=full_prompt)
    print("Bot Text Response:", response_text)

    # Update conversation memory
    conversation_memory.append(f"User: {user_input}")
    conversation_memory.append(f"Bot: {response_text}")

    # =========================
    # 3. Check if image requested
    # =========================
    def needs_image(text):
        triggers = ["selfie", "picture", "photo", "draw me", "show me", "illustrate"]
        return any(word in text.lower() for word in triggers)

    if needs_image(user_input):
        # Create image prompt using both bot identity and conversation context
        image_prompt = f"{bot_identity_description}, situation: {user_input}, previous context: {response_text}"

        # Generate image
        generated_image = image_model.generate(prompt=image_prompt)

        # Return text + image
        print("Bot Image Response:", generated_image)