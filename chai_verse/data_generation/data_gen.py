#!/usr/bin/env python3
import json, random, requests, time, re
from flags import get_args
from constants import *
from tqdm import tqdm

CATS = [
    "relationship","roleplay","roleplay_relationship","fantasy",
    "mafia","mafia_relationship","mafia_celebrity"
]

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

# Persona prompts (character vs. fantasy)
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
    "- If the bot_name is a location, object, or land (e.g., Floating Sky Castle), describe it as a PLACE or ENTITY with atmosphere, lore, and mysteries.\n"
    "- Focus on environment, mood, history, and what the USER experiences there.\n"
    "- Do not assign human-like relationships (no 'girlfriend', 'boyfriend').\n"
    "- Keep it immersive and believable as part of a fantasy world.\n"
    "- Output ONLY the description, no extra commentary.\n"
    "- Bot name: {bot_name}\n"
    "- Category: {category}\n"
)

# First-message opener (bot “reaches out” 50% of the time; never for fantasy)
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

JUDGE_RUBRIC = (
    "You are a strict judge evaluating roleplay chatbot responses from the USER’s perspective. "
    "Pick the answer that is most interesting, engaging, emotionally satisfying, and specific to the seed/persona. "
    "Prefer 35–65 words; avoid short or rambling replies. "
    "Reply with ONLY the number (1, 2, or 3)."
)

# ==============================
# Helpers
# ==============================
def normalize_probs(p):
    total = sum(p.values())
    return {k: (v/total if total > 0 else 0) for k,v in p.items()}

def pick_relationship(gender: str) -> str:
    if gender == "female": return random.choice(RELATIONSHIP_FEMALE)
    elif gender == "male": return random.choice(RELATIONSHIP_MALE)
    else: return random.choice(RELATIONSHIP_FEMALE if random.random() < 0.5 else RELATIONSHIP_MALE)

def assign_gender_from_archetype(tag: str) -> str:
    if tag in ("male","female"): return tag
    return "female" if random.random() < 0.5 else "male"

def call(endpoint, model_name, messages, max_tokens=256, temperature=0.9, top_p=0.95):
    r = requests.post(endpoint, json={
        "model": model_name,"messages": messages,
        "temperature": temperature,"top_p": top_p,"max_tokens": max_tokens
    }, timeout=300)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def count_words(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s))

def clamp_words(s: str, max_words=65) -> str:
    words = re.findall(r"\S+", s.strip())
    if len(words) <= max_words:
        return s.strip()
    return " ".join(words[:max_words]).strip()

def extract_last_bot_line(block: str, bot_name: str) -> str:
    lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
    last = ""
    prefix = f"{bot_name}:"
    for ln in lines:
        if ln.startswith(prefix):
            last = ln
    if last.startswith(prefix):
        return last[len(prefix):].strip()
    return last

def ensure_alternation(block: str, bot_name: str, turns: int) -> str:
    lines = [ln for ln in [x.strip() for x in block.strip().splitlines()] if ln]
    good = []
    bot_pref = f"{bot_name}:"
    for ln in lines:
        if ln.startswith(bot_pref) or ln.startswith("User:"):
            good.append(ln)
    if not good or not good[0].startswith(bot_pref):
        good.insert(0, f"{bot_name}: ...")
    if len(good) > turns:
        good = good[:turns]
    if not good[-1].startswith(bot_pref):
        for ln in reversed(lines):
            if ln.startswith(bot_pref):
                good[-1] = ln
                break
    if len(good) > turns:
        good = good[:turns]
    return "\n".join(good)

def format_pygmalion(user_text_block, final_bot_reply, bot_name, persona_header, persona_spoken):
    # NOTE: Keeping "BotName's Persona:" header (you can set persona_header="" to leave blank)
    persona_hdr = f"{bot_name}'s Persona: {persona_header}\n####\n"
    conv = persona_hdr
    conv += f"{bot_name}: {persona_spoken}\n"
    conv += f"{user_text_block}\n"
    conv += f"{bot_name}: {final_bot_reply}\n<START>\n"
    return {"prompt": conv, "response": f"{bot_name}: {final_bot_reply}"}

# Expand / tighten the final line to be within 35–65 words, keep voice
SYSTEM_TIGHTEN_FMT = (
    "You carefully rewrite the BOT'S CLOSING LINE to be between 35 and 65 words, preserving tone and intent.\n"
    "Rules:\n"
    "- Output ONLY the rewritten sentence (no labels like 'Bot:' or 'User:').\n"
    "- Keep the same voice and persona.\n"
    "- Avoid adding generic filler; keep it vivid and specific.\n"
    "Bot name: {bot_name}\n"
    "Persona:\n{persona}\n"
)
def ensure_final_range(gen_url, gen_model, bot_name, persona_spoken, final_line, min_w=35, max_w=65):
    w = count_words(final_line)
    if w < min_w:
        sys = SYSTEM_TIGHTEN_FMT.format(bot_name=bot_name, persona=persona_spoken)
        out = call(
            gen_url, gen_model,
            [
                {"role":"system","content":sys},
                {"role":"user","content":f"Original closing line:\n{final_line}\n\nRewrite now within {min_w}-{max_w} words."}
            ],
            max_tokens=160, temperature=0.8, top_p=0.95
        ).strip()
        # If the model accidentally includes a label, strip it
        out = re.sub(rf"^{re.escape(bot_name)}\s*:\s*", "", out).strip()
        # If still too long, clamp softly
        if count_words(out) > max_w:
            out = clamp_words(out, max_words=max_w)
        return out
    if w > max_w:
        return clamp_words(final_line, max_words=max_w)
    return final_line

# ==============================
# Persona & Opener generation
# ==============================
def generate_persona(gen_url, gen_model, bot_name, category, seed):
    if category == "fantasy":
        sys = SYSTEM_PERSONA_FMT_FANTASY.format(bot_name=bot_name, category=category)
        user_msg = f"Seed: {seed}\nWrite the fantasy scenario description."
    else:
        sys = SYSTEM_PERSONA_FMT_CHARACTER.format(bot_name=bot_name, category=category)
        user_msg = f"Seed: {seed}\nWrite the persona."

    persona = call(
        gen_url, gen_model,
        [
            {"role": "system", "content": sys},
            {"role": "user", "content": user_msg}
        ],
        max_tokens=180, temperature=1.0, top_p=0.95
    ).strip()
    return persona

def generate_opener(gen_url, gen_model, bot_name, category, persona_text):
    sys = SYSTEM_OPENER_FMT.format(bot_name=bot_name, category=category, persona=persona_text)
    opener = call(
        gen_url, gen_model,
        [
            {"role": "system", "content": sys},
            {"role": "user", "content": "Write the single first outreach line now."}
        ],
        max_tokens=40, temperature=0.9, top_p=0.95
    ).strip()
    opener = re.sub(r'\s+', ' ', opener).strip()
    # Never add speaker labels here; we append to persona line
    opener = re.sub(rf"^{re.escape(bot_name)}\s*:\s*", "", opener).strip()
    return opener

# ==============================
# Category → bot name
# ==============================
def build_bot_name(category: str):
    if category == "fantasy":
        return random.choice(FANTASY_SCENARIOS)
    if category == "relationship":
        gender="female" if random.random()<0.5 else "male"
        return f"{random.choice(FIRST_NAMES)} {pick_relationship(gender)}"
    if category == "roleplay":
        role,tag=random.choice(ROLEPLAY_ARCHETYPES)
        return f"{random.choice(FIRST_NAMES)} {role}"
    if category == "roleplay_relationship":
        role,tag=random.choice(ROLEPLAY_ARCHETYPES)
        gender=assign_gender_from_archetype(tag)
        rel=pick_relationship(gender)
        return f"{random.choice(FIRST_NAMES)} {role} {rel}"
    if category == "mafia":
        role,tag=random.choice(MAFIA_ARCHETYPES)
        return f"{random.choice(FIRST_NAMES)} {role}"
    if category == "mafia_relationship":
        role,tag=random.choice(MAFIA_ARCHETYPES)
        rel=pick_relationship(tag)
        return f"{random.choice(FIRST_NAMES)} {role} {rel}"
    if category == "mafia_celebrity":
        celeb=random.choice(CELEBRITIES_FEMALE if random.random()<0.5 else CELEBRITIES_MALE)
        role,tag=random.choice(MAFIA_ARCHETYPES)
        return f"{celeb} {role}"
    return "Bot"

# ==============================
# Conversation generation (block)
# ==============================
def generate_block(gen_url, gen_model, bot_name, persona_spoken, seed_for_user, turns):
    sys = SYSTEM_BLOCK_FMT.format(bot=bot_name, persona_spoken=persona_spoken, turns=turns)
    user_instr = (
        f"SEED CONTEXT:\n{seed_for_user}\n\n"
        f"PERSONA (first line must be exactly this speaker):\n{bot_name}: {persona_spoken}\n\n"
        f"Now produce the conversation block with EXACTLY {turns} lines, alternating '{bot_name}:' and 'User:' "
        f"and ENDING with '{bot_name}:' only. The final '{bot_name}:' line MUST be 35–65 words. No extra commentary."
    )
    drafts = []
    for _ in range(3):
        out = call(gen_url, gen_model, [
            {"role":"system","content":sys},
            {"role":"user","content":user_instr}
        ], max_tokens=640, temperature=0.9, top_p=0.95).strip()
        out = ensure_alternation(out, bot_name, turns)
        drafts.append(out)
    return drafts

def judge_best_block(seed_prompt, blocks, judge_url, judge_model, bot_name):
    finals, kept_idx = [], []
    for i, b in enumerate(blocks):
        if not isinstance(b, str) or not b.strip():
            continue
        last = extract_last_bot_line(b, bot_name)
        if last:
            finals.append(last)
            kept_idx.append(i)

    if not finals:
        return 0, f"{bot_name}: ...", []

    numbered = "\n\n".join([f"{i+1}) {t}" for i, t in enumerate(finals)])
    try:
        decision = call(
            judge_url, judge_model,
            [
                {"role": "system", "content": JUDGE_RUBRIC},
                {"role": "user", "content": f"Seed: {seed_prompt}\n\nFinal lines:\n{numbered}\n\nBest:"}
            ],
            max_tokens=5, temperature=0.0, top_p=1.0
        )
    except Exception as e:
        print(f"[WARN] Judge model call failed: {e}")
        return kept_idx[0], finals[0], finals[1:]

    winner_local = None
    for ch in decision:
        if ch in "123456789":
            cand = int(ch) - 1
            if 0 <= cand < len(finals):
                winner_local = cand
                break

    if winner_local is None:
        print(f"[WARN] Unexpected judge reply: {decision!r}, defaulting to first candidate")
        winner_local = 0

    winner_global = kept_idx[winner_local]
    final_line = finals[winner_local]
    rejected_finals = [f for j, f in enumerate(finals) if j != winner_local]
    return winner_global, final_line, rejected_finals

# ==============================
# Main
# ==============================
def main():
    args = get_args()
    if args.single_server:
        args.judge_url=args.gen_url
        args.judge_model_name=args.gen_model_name

    # Category probabilities
    raw_probs={
        "relationship":args.prob_relationship,"roleplay":args.prob_roleplay,
        "roleplay_relationship":args.prob_roleplay_relationship,"fantasy":args.prob_fantasy,
        "mafia":args.prob_mafia,"mafia_relationship":args.prob_mafia_relationship,
        "mafia_celebrity":args.prob_mafia_celebrity
    }
    probs=normalize_probs(raw_probs)
    cat_bins=[]; acc=0.0
    for c in CATS: acc+=probs[c]; cat_bins.append((c,acc))
    def draw_cat():
        r=random.random()
        for c,u in cat_bins:
            if r<=u: return c
        return CATS[-1]

    # === Example mode (pretty-print ONE item; no files written) ===
    if getattr(args, "example", False):
        target_turns = 3
        category = draw_cat()
        bot_name = build_bot_name(category)

        seed_text = "seed"
        if args.persona:
            persona_header = args.persona
            persona_spoken = args.persona
        else:
            persona_header = ""
            persona_spoken = generate_persona(args.gen_url, args.gen_model_name, bot_name, category, seed_text)

        # 50% opener ONLY for non-fantasy and if persona not user-specified
        if category != "fantasy" and not args.persona and random.random() < 0.5:
            try:
                opener = generate_opener(args.gen_url, args.gen_model_name, bot_name, category, persona_spoken)
                if opener:
                    persona_spoken = (persona_spoken.rstrip() + " " + opener.strip()).strip()
            except Exception as e:
                print(f"[WARN] opener generation failed: {e}")

        user_seed = persona_spoken
        blocks = generate_block(args.gen_url, args.gen_model_name, bot_name, persona_spoken, user_seed, target_turns)
        winner_idx, final_line, rejected_finals = judge_best_block(user_seed, blocks, args.judge_url, args.judge_model_name, bot_name)

        if winner_idx is None:
            print("\n[WARN] Example skipped (no valid candidates).")
            return

        final_line = ensure_final_range(args.gen_url, args.gen_model_name, bot_name, persona_spoken, final_line, 35, 65)

        chosen_block_lines = blocks[winner_idx].strip().splitlines()
        if chosen_block_lines and chosen_block_lines[-1].startswith(f"{bot_name}:"):
            chosen_body = "\n".join(chosen_block_lines[:-1])
        else:
            chosen_body = "\n".join(chosen_block_lines)

        sft_ex = format_pygmalion(chosen_body, final_line, bot_name, persona_header, persona_spoken)
        dpo_ex = {
            "prompt": chosen_body,
            "chosen": final_line,
            "rejected": [clamp_words(x, max_words=65) for x in rejected_finals]
        }

        print(f"\n=== Example Category === {category}")
        print("\n=== SFT Example ===")
        print(json.dumps(sft_ex, indent=2, ensure_ascii=False))
        print("\n=== DPO Example ===")
        print(json.dumps(dpo_ex, indent=2, ensure_ascii=False))
        return

    # Normal mode
    if not args.out: raise ValueError("--out is required")

    total_resp_words=0; min_resp_words=10**9; max_resp_words=0; last_resp_words=0
    total_turns=0; min_turns=10**9; max_turns=0
    skipped=0
    start=time.time()

    sft_path=f"{args.out}_sft.jsonl"; dpo_path=f"{args.out}_dpo.jsonl"
    with open(sft_path,"w") as sft_out, open(dpo_path,"w") as dpo_out:
        for idx in range(1, args.n_prompts+1):
            target_turns = 3
            category = draw_cat()
            bot_name = build_bot_name(category)

            seed_text = "seed"
            if args.persona:
                persona_header = args.persona
                persona_spoken = args.persona
            else:
                persona_header = ""
                persona_spoken = generate_persona(args.gen_url, args.gen_model_name, bot_name, category, seed_text)

            if category != "fantasy" and not args.persona and random.random() < 0.5:
                try:
                    opener = generate_opener(args.gen_url, args.gen_model_name, bot_name, category, persona_spoken)
                    if opener:
                        persona_spoken = (persona_spoken.rstrip() + " " + opener.strip()).strip()
                except Exception as e:
                    print(f"[WARN] opener generation failed: {e}")

            user_seed = persona_spoken
            blocks = generate_block(args.gen_url, args.gen_model_name, bot_name, persona_spoken, user_seed, target_turns)
            winner_idx, final_line, rejected_finals = judge_best_block(user_seed, blocks, args.judge_url, args.judge_model_name, bot_name)

            if winner_idx is None:
                skipped += 1
                continue

            final_line = ensure_final_range(args.gen_url, args.gen_model_name, bot_name, persona_spoken, final_line, 35, 65)

            chosen_block_lines = blocks[winner_idx].strip().splitlines()
            if chosen_block_lines and chosen_block_lines[-1].startswith(f"{bot_name}:"):
                chosen_body = "\n".join(chosen_block_lines[:-1])
            else:
                chosen_body = "\n".join(chosen_block_lines)

            formatted = format_pygmalion(chosen_body, final_line, bot_name, persona_header, persona_spoken)
            sft_out.write(json.dumps(formatted)+"\n"); sft_out.flush()

            for rej in rejected_finals:
                dpo_out.write(json.dumps({
                    "prompt": chosen_body,
                    "chosen": final_line,
                    "rejected": clamp_words(rej, max_words=65)
                })+"\n")
                dpo_out.flush()

            turns = formatted["prompt"].count("User:") + formatted["prompt"].count(f"{bot_name}:")
            total_turns += turns; min_turns=min(min_turns,turns); max_turns=max(max_turns,turns)

            last_resp_words = count_words(final_line)
            total_resp_words += last_resp_words
            min_resp_words = min(min_resp_words,last_resp_words)
            max_resp_words = max(max_resp_words,last_resp_words)

            if idx % 10 == 0 or idx == args.n_prompts:
                kept = idx - skipped
                elapsed=time.time()-start
                avg = elapsed/max(1,kept)
                eta=(args.n_prompts-idx)*avg/60
                avg_resp = total_resp_words/max(1,kept)
                print(f"[{idx}/{args.n_prompts}] kept={kept}, skipped={skipped}  {avg:.2f}s/kept example, ETA {eta:.1f}m")
                print(f"  Avg final line: {avg_resp:.1f} words (min={min_resp_words}, max={max_resp_words}, last={last_resp_words})")
                print(f"  Avg turns: {total_turns/max(1,kept):.2f} (min={min_turns}, max={max_turns})")

                # Debug peeks
                print("  === Sample SFT ===")
                try:
                    print(json.dumps(formatted, indent=2, ensure_ascii=False)[:500])
                except Exception as e:
                    print(f"    [WARN] Could not print SFT sample: {e}")

                print("  === Sample DPO ===")
                try:
                    if rejected_finals:
                        dpo_sample={
                            "prompt":chosen_body,
                            "chosen":final_line,
                            "rejected":clamp_words(rejected_finals[0], max_words=65)
                        }
                        print(json.dumps(dpo_sample, indent=2, ensure_ascii=False)[:500])
                    else:
                        print("    [No rejected candidates]")
                except Exception as e:
                    print(f"    [WARN] Could not print DPO sample: {e}")

    print(f"Done. Wrote {sft_path} and {dpo_path}. Skipped {skipped} examples.")

if __name__=="__main__":
    main()
