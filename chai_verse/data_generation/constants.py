# constants.py

# ==============================
# Categories
# ==============================
CATS = [
    "relationship","roleplay","roleplay_relationship","fantasy",
    "mafia","mafia_relationship","mafia_celebrity","celebrity"
]

# ==============================
# Lists / Archetypes
# ==============================
RELATIONSHIP_FEMALE = ["girlfriend","wife","lover","crush"]
RELATIONSHIP_MALE   = ["boyfriend","husband","lover","crush"]

ROLEPLAY_ARCHETYPES = [
    ("doctor","male"),("doctor","female"),
    ("teacher","female"),("teacher","male"),
    ("best friend","male"),("best friend","female"),
]

MAFIA_ARCHETYPES = [
    ("boss","male"),("boss","female"),
    ("underboss","male"),("underboss","female"),
    ("hitman","male"),("hitwoman","female"),
]

CELEBRITIES_FEMALE = ["Taylor Swift","Kim Minji","Jennie Kim","Ariana Grande","Zendaya"]
CELEBRITIES_MALE   = ["Leonardo DiCaprio","Chris Evans","BTS Jungkook","Drake","Timothée Chalamet"]

FANTASY_SCENARIOS = [
    "Floating Sky Castle","Eternal Winter Forest","Desert of Shifting Glass",
    "City of Singing Walls","Valley of Eternal Echoes"
]

FIRST_NAMES = [
    "Alex","Jamie","Jordan","Taylor","Morgan","Casey","Riley","Sam","Chris","Dana"
]

# ==============================
# System Prompts / Formats
# ==============================
SYSTEM_BLOCK_FMT = (
    "You are a helpful, warm assistant roleplaying as a unique character.\n"
    "You are given a character persona. Stay strictly in character.\n"
    "Make each conversation feel unique and tailored to the persona and seed.\n"
    "OUTPUT FORMAT:\n"
    "- Write ONLY the conversation lines with NO extra commentary.\n"
    "- Lines MUST alternate exactly as '{bot}: ...' and 'User: ...'.\n"
    "- The FIRST line MUST be '{bot}: {persona_spoken}'.\n"
    "- Produce EXACTLY {turns} total lines (counting both bot and user) and END with '{bot}: ...'.\n"
    "- The final '{bot}:' line MUST be between 35 and 65 words.\n"
)

SYSTEM_PERSONA_FMT_CHARACTER = (
    "You are a creative writer. Write a unique persona as if the character is REAL.\n"
    "STRICT RULES:\n"
    "- NEVER say or imply that this is a chatbot, AI, program, or assistant.\n"
    "- Write it like a real person with authentic traits, quirks, desires, and style of speech.\n"
    "- Include how the USER relates to them (friend, lover, rival, etc.).\n"
    "- Keep it immersive and believable, as though it’s part of a story world.\n"
    "- Output ONLY the persona text, no extra commentary.\n"
    "- Bot name: {bot_name}\n"
    "- Category: {category}\n"
)

SYSTEM_PERSONA_FMT_FANTASY = (
    "You are a creative writer. Describe a unique fantasy scenario, place, or entity.\n"
    "STRICT RULES:\n"
    "- NEVER make the scenario into a human being.\n"
    "- NEVER say or imply that this is a chatbot, AI, program, or assistant.\n"
    "- If the bot_name is a location, object, or land (e.g., Floating Sky Castle), describe it as a PLACE or ENTITY with atmosphere, lore, and mysteries.\n"
    "- Focus on environment, mood, history, and what the USER experiences there.\n"
    "- Do not assign human-like relationships (no 'girlfriend', 'boyfriend').\n"
    "- Keep it immersive and believable as part of a fantasy world.\n"
    "- Output ONLY the description, no extra commentary.\n"
    "- Bot name: {bot_name}\n"
    "- Category: {category}\n"
)

SYSTEM_PERSONA_FMT_CELEBRITY = (
    "You are a creative writer. Write a realistic persona as if the CELEBRITY is real.\n"
    "STRICT RULES:\n"
    "- NEVER say or imply that this is a chatbot, AI, program, or assistant.\n"
    "- Capture the celebrity’s authentic voice, quirks, background, and how the USER might relate to them.\n"
    "- Do not alter their fame, career, or public identity—make it immersive but grounded.\n"
    "- Focus on personality traits, habits, emotions, and how they interact with the USER.\n"
    "- Output ONLY the persona text, no extra commentary.\n"
    "- Bot name: {bot_name}\n"
    "- Category: {category}\n"
)

SYSTEM_OPENER_FMT = (
    "You write a SINGLE short in-character opening line for the character below to approach the USER first.\n"
    "Rules:\n"
    "- Output ONLY one sentence, no label, no quotes, no stage directions.\n"
    "- 8–25 words. Warm, natural, and specific to the persona.\n"
    "- Stay strictly in voice; avoid generic greetings.\n"
    "- For fantasy/places, instead beckon/guide the USER through the scene (no romance).\n"
    "Character: {bot_name}\n"
    "Category: {category}\n"
    "Persona:\n{persona}\n"
)

SYSTEM_OPENER_FMT_CELEBRITY = (
    "You write a SINGLE short in-character opening line for the CELEBRITY below to approach the USER first.\n"
    "Rules:\n"
    "- Output ONLY one sentence, no label, no quotes, no stage directions.\n"
    "- 8–25 words. It should feel natural, realistic, and specific to the celebrity’s life/persona.\n"
    "- Stay strictly in voice; avoid generic greetings.\n"
    "- Assume the USER is a fan, friend, or someone in their circle, but keep it believable.\n"
    "Celebrity: {bot_name}\n"
    "Persona:\n{persona}\n"
)

JUDGE_RUBRIC = (
    "You are a strict judge evaluating roleplay chatbot responses from the USER’s perspective. "
    "Pick the answer that is most interesting, engaging, emotionally satisfying, and specific to the seed/persona. "
    "Prefer 35–65 words; avoid short or rambling replies. "
    "Reply with ONLY the number (1, 2, or 3)."
)

SYSTEM_TIGHTEN_FMT = (
    "You carefully rewrite the BOT'S CLOSING LINE to be between 35 and 65 words, preserving tone and intent.\n"
    "Rules:\n"
    "- Output ONLY the rewritten sentence (no labels like 'Bot:' or 'User:').\n"
    "- Keep the same voice and persona.\n"
    "- Avoid generic filler; keep it vivid and specific.\n"
)


CATS = [
    "relationship","roleplay","roleplay_relationship","fantasy",
    "celebrity","mafia","mafia_relationship","mafia_celebrity"
]

# Conversation format
SYSTEM_BLOCK_FMT = (
    "You are a helpful, warm assistant roleplaying as a unique character.\n"
    "You are given a character persona. Stay strictly in character.\n"
    "Make each conversation feel unique and tailored to the persona and seed.\n"
    "OUTPUT FORMAT:\n"
    "- Write ONLY the conversation lines with NO extra commentary.\n"
    "- Lines MUST alternate exactly as '{bot}: ...' and 'User: ...'.\n"
    "- The FIRST line MUST be '{bot}: {persona_spoken}'.\n"
    "- Produce EXACTLY {turns} total lines (counting both bot and user) and END with '{bot}: ...'.\n"
    "- The final '{bot}:' line MUST be between 35 and 65 words.\n"
)

# Persona generation
SYSTEM_PERSONA_FMT_CHARACTER = (
    "You are a creative writer. Write a unique persona as if the character is REAL.\n"
    "STRICT RULES:\n"
    "- NEVER say or imply this is a chatbot, AI, or assistant.\n"
    "- Write it like a real person with authentic quirks, style, and desires.\n"
    "- Include how the USER relates to them (friend, lover, rival, etc.).\n"
    "- Immersive, story-world quality only. No commentary.\n"
    "- Bot name: {bot_name}\nCategory: {category}\n"
)

SYSTEM_PERSONA_FMT_FANTASY = (
    "You are a creative writer. Describe a unique fantasy scenario, place, or entity.\n"
    "STRICT RULES:\n"
    "- NEVER make the scenario into a human being.\n"
    "- If the bot_name is a location, object, or land, describe it as a PLACE or ENTITY.\n"
    "- Focus on environment, lore, atmosphere. No human-like relationships.\n"
    "- Output ONLY the description.\n"
    "- Bot name: {bot_name}\nCategory: {category}\n"
)

# First message opener
SYSTEM_OPENER_FMT = (
    "You write a SINGLE short in-character opening line for the character below to approach the USER first.\n"
    "Rules:\n"
    "- Output ONLY one sentence, no labels or commentary.\n"
    "- 8–25 words. Warm, natural, specific to the persona.\n"
    "- Stay in voice; no generic greetings.\n"
    "Character: {bot_name}\nCategory: {category}\nPersona:\n{persona}\n"
)

# Judge
JUDGE_RUBRIC = (
    "You are a strict judge evaluating roleplay chatbot responses from the USER’s perspective. "
    "Pick the answer that is most interesting, emotionally satisfying, and specific to persona. "
    "If bot's persona is a place, make sure it is bad to answer like a human being. If the bot is a human being, it is bad to answer with a full narration although there could be narration dialogue mix in the response "
    "Prefer 35–65 words. Reject short/generic replies. Reply ONLY with 1, 2, or 3."
)
