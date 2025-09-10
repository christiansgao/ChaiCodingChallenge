#!/usr/bin/env python3
import json, random, requests, time, re
from flags import get_args
from constants import *
from tqdm import tqdm

# ==============================
# Helpers
# ==============================
def normalize_probs(p):
    total = sum(p.values())
    return {k: (v/total if total > 0 else 0) for k,v in p.items()}

def pick_relationship(gender: str) -> str:
    if gender == "female": return random.choice(RELATIONSHIP_FEMALE)
    elif gender == "male": return random.choice(RELATIONSHIP_MALE)
    else: return random.choice(RELATIONSHIP_FEMALE if random.random()<0.5 else RELATIONSHIP_MALE)

def assign_gender_from_archetype(tag: str) -> str:
    if tag in ("male","female"): return tag
    return "female" if random.random()<0.5 else "male"

def call(endpoint, model_name, messages, max_tokens=256, temperature=0.9, top_p=0.95):
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens
    }
    try:
        r = requests.post(endpoint, json=payload, timeout=300)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print("[ERROR] Bad request to", endpoint)
        print("Payload:", json.dumps(payload, indent=2))
        print("Response:", getattr(r, "text", None))
        raise

def count_words(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s))

def clamp_words(s: str, max_words=65) -> str:
    words = re.findall(r"\S+", s.strip())
    return " ".join(words[:max_words]).strip()

def extract_last_bot_line(block: str, bot_name: str) -> str:
    lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
    prefix=f"{bot_name}:"
    last=""
    for ln in lines:
        if ln.startswith(prefix): last=ln
    return last[len(prefix):].strip() if last.startswith(prefix) else last

def ensure_alternation(block: str, bot_name: str, turns: int) -> str:
    lines=[ln for ln in [x.strip() for x in block.strip().splitlines()] if ln]
    good=[ln for ln in lines if ln.startswith(f"{bot_name}:") or ln.startswith("User:")]
    if not good or not good[0].startswith(f"{bot_name}:"): good.insert(0, f"{bot_name}: ...")
    if len(good)>turns: good=good[:turns]
    if not good[-1].startswith(f"{bot_name}:"):
        for ln in reversed(lines):
            if ln.startswith(f"{bot_name}:"): good[-1]=ln; break
    return "\n".join(good[:turns])

def clean_block_text(text: str) -> str:
    # Replace narration [ ... ] with italics * ... *
    text = re.sub(r"\[(.*?)\]", r"*\1*", text)
    # Collapse duplicate newlines
    text = re.sub(r"\n\s*\n", "\n", text)
    return text.strip()

def format_pygmalion(user_text_block, final_bot_reply, bot_name, persona_header, persona_spoken):
    persona_hdr = f"{bot_name}'s Persona: {persona_header}\n####\n"
    conv = persona_hdr
    conv += f"{bot_name}: {persona_spoken}\n"
    if user_text_block.strip():
        conv += clean_block_text(user_text_block) + "\n"
    # ⚠️ Do NOT append the final bot line into the prompt
    conv += "<START>\n"
    return {
        "prompt": conv,
        "response": f"{bot_name}: {clean_block_text(final_bot_reply)}"
    }

# === NEW: split body vs final bot line (so SFT doesn't duplicate) ===
def split_block_and_last_bot(block: str, bot_name: str):
    """
    Return (body_without_first_bot_and_last_bot, final_bot_text_without_prefix).
    Ensures body starts after the FIRST bot line and ends before the LAST bot line.
    """
    lines = [ln for ln in (block or "").splitlines() if ln.strip()]
    bot_pref = f"{bot_name}:"

    # locate first bot line (persona_spoken) and last bot line
    first_bot_idx = None
    last_bot_idx = None
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith(bot_pref):
            first_bot_idx = i
            break
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].lstrip().startswith(bot_pref):
            last_bot_idx = i
            break

    if last_bot_idx is None:
        return "\n".join(lines[first_bot_idx+1:] if first_bot_idx is not None else lines), ""

    final_bot_text = lines[last_bot_idx].lstrip()[len(bot_pref):].strip()

    # body excludes the first bot line and the final bot line
    start = (first_bot_idx + 1) if first_bot_idx is not None else 0
    body_lines = lines[start:last_bot_idx]

    # ensure body ends on a User line (trim stray trailing bot)
    while body_lines and body_lines[-1].lstrip().startswith(bot_pref):
        body_lines.pop()

    return "\n".join(body_lines), final_bot_text

# Expand to 35–65 words if needed
def ensure_final_range(gen_url, gen_model, bot_name, persona_spoken, final_line, min_w=35, max_w=65):
    w = count_words(final_line)
    if w < min_w or w > max_w:
        sys = SYSTEM_TIGHTEN_FMT.format(bot_name=bot_name, persona=persona_spoken)
        raw = call(
            gen_url, gen_model,
            [
                {"role": "system", "content": sys},
                {"role": "user", "content": f"Original closing line:\n{final_line}\n\nRewrite only the line in {min_w}-{max_w} words. Output ONLY the text, no extra commentary."}
            ],
            max_tokens=200, temperature=0.8, top_p=0.95
        ).strip()
        # Clean rewrite: strip labels and "Sure..." preamble
        out = re.sub(rf"^{re.escape(bot_name)}\s*:\s*", "", raw).strip()
        out = re.sub(r"^(Sure, here is.*?:)", "", out, flags=re.I).strip()
        if not out:
            return final_line  # fallback
        if count_words(out) > max_w:
            out = clamp_words(out, max_words=max_w)
        return out
    return final_line

# ==============================
# Persona & Opener generation
# ==============================
def generate_persona(gen_url, gen_model, bot_name, category, seed):
    if category=="fantasy":
        sys=SYSTEM_PERSONA_FMT_FANTASY.format(bot_name=bot_name,category=category)
        user=f"Seed: {seed}\nWrite the fantasy scenario description."
    else:
        sys=SYSTEM_PERSONA_FMT_CHARACTER.format(bot_name=bot_name,category=category)
        user=f"Seed: {seed}\nWrite the persona."
    return call(gen_url, gen_model,[{"role":"system","content":sys},{"role":"user","content":user}],
                max_tokens=180, temperature=1.0, top_p=0.95).strip()

def generate_opener(gen_url, gen_model, bot_name, category, persona_text):
    sys=SYSTEM_OPENER_FMT.format(bot_name=bot_name,category=category,persona=persona_text)
    opener=call(gen_url, gen_model,[{"role":"system","content":sys},{"role":"user","content":"Write it now."}],
                max_tokens=40, temperature=0.9, top_p=0.95).strip()
    opener=re.sub(rf"^{re.escape(bot_name)}\s*:\s*","",opener).strip()
    return re.sub(r"\s+"," ",opener)

# ==============================
# Category → bot name
# ==============================
def build_bot_name(category:str):
    if category=="fantasy": return random.choice(FANTASY_SCENARIOS)
    if category=="relationship":
        gender="female" if random.random()<0.5 else "male"
        return f"{random.choice(FIRST_NAMES)} {pick_relationship(gender)}"
    if category=="roleplay":
        role,tag=random.choice(ROLEPLAY_ARCHETYPES)
        return f"{random.choice(FIRST_NAMES)} {role}"
    if category=="roleplay_relationship":
        role,tag=random.choice(ROLEPLAY_ARCHETYPES)
        gender=assign_gender_from_archetype(tag); rel=pick_relationship(gender)
        return f"{random.choice(FIRST_NAMES)} {role} {rel}"
    if category=="celebrity":
        return random.choice(CELEBRITIES_FEMALE+CELEBRITIES_MALE)
    if category=="mafia":
        role,tag=random.choice(MAFIA_ARCHETYPES)
        return f"{random.choice(FIRST_NAMES)} {role}"
    if category=="mafia_relationship":
        role,tag=random.choice(MAFIA_ARCHETYPES); rel=pick_relationship(tag)
        return f"{random.choice(FIRST_NAMES)} {role} {rel}"
    if category=="mafia_celebrity":
        celeb=random.choice(CELEBRITIES_FEMALE+CELEBRITIES_MALE); role,tag=random.choice(MAFIA_ARCHETYPES)
        return f"{celeb} {role}"
    return "Bot"

# ==============================
# Turn-by-turn conversation generation (iterative)
# ==============================
def _gen_next_line(gen_url, gen_model, bot_name, persona_spoken, prior_lines, want_role, is_final=False):
    """
    want_role: 'bot' or 'user'
    is_final: if True AND want_role='bot', enforce 35–65 words on this line.
    """
    history = "\n".join(prior_lines)
    if want_role == "bot":
        sys = (
                "Write EXACTLY ONE next line for the BOT.\n"
                f"- Label it as '{bot_name}: ...'\n"
                "- Stay strictly in character and follow the established style.\n"
                "- No stage directions outside italics *like this*.\n"
                + ("- This is the FINAL bot line; make it 35–65 words, vivid and polished.\n" if is_final else "")
                + "- Output only that single line."
        )
        user = f"Conversation so far:\n{history}\n\nWrite the next bot line now."
    else:
        sys = (
            "Write EXACTLY ONE next line for the USER.\n"
            "- Label it as 'User: ...'\n"
            "- Keep it natural and reactive. No rambling.\n"
            "- Output only that single line."
        )
        user = f"Conversation so far:\n{history}\n\nWrite the next user line now."

    out = call(
        gen_url, gen_model,
        [
            {"role": "system", "content": sys},
            {"role": "user", "content": user}
        ],
        max_tokens=200, temperature=0.9, top_p=0.95
    ).strip()

    # hard sanitize to exactly one labeled line
    label = f"{bot_name}:" if want_role=="bot" else "User:"
    # take only the first labeled line that starts correctly
    first_line = ""
    for ln in [x.strip() for x in out.splitlines() if x.strip()]:
        if ln.startswith(label):
            first_line = ln
            break
    if not first_line:
        # fallback: force a skeleton
        first_line = f"{label} ..."

    # clean narration: [ ... ] -> * ... *
    first_line = re.sub(r"\[(.*?)\]", r"*\1*", first_line)
    # if final bot line needs tightening, we handle later globally
    return first_line

def generate_block_iterative(gen_url, gen_model, bot_name, persona_spoken, turns):
    """
    Build a conversation of `turns` lines iteratively, starting with:
      1) Bot: {persona_spoken}
      2) User: ...
      3) Bot: ...
      ... ending with Bot.
    """
    lines = [f"{bot_name}: {persona_spoken}"]
    # We want alternating: bot, user, bot, ... and end with bot
    for i in range(2, turns + 1):
        is_final = (i == turns)
        want_role = "user" if i % 2 == 0 else "bot"
        ln = _gen_next_line(gen_url, gen_model, bot_name, persona_spoken, lines, want_role, is_final and want_role=="bot")
        lines.append(ln)
    return "\n".join(lines)

def generate_block(gen_url, gen_model, bot_name, persona_spoken, seed_for_user, turns):
    # wrapper to preserve function name/signature; use the iterative generator
    return [generate_block_iterative(gen_url, gen_model, bot_name, persona_spoken, turns) for _ in range(3)]

def judge_best_block(seed_prompt, blocks, judge_url, judge_model, bot_name):
    finals,idxs=[],[]
    for i,b in enumerate(blocks):
        if not b.strip(): continue
        last=extract_last_bot_line(b,bot_name)
        if last: finals.append(last); idxs.append(i)
    if not finals: return None,"",[]
    numbered="\n\n".join([f"{i+1}) {t}" for i,t in enumerate(finals)])
    decision=call(judge_url,judge_model,[{"role":"system","content":JUDGE_RUBRIC},
                {"role":"user","content":f"Seed: {seed_prompt}\n\nFinal lines:\n{numbered}\n\nBest:"}],
                max_tokens=5,temperature=0.0,top_p=1.0)
    for ch in decision:
        if ch in "123" and int(ch)-1<len(finals):
            return idxs[int(ch)-1], finals[int(ch)-1], [f for j,f in enumerate(finals) if j!=int(ch)-1]
    print(f"[WARN] Judge bad reply {decision!r}, fallback to first.")
    return idxs[0], finals[0], finals[1:]

# ==============================
# Main
# ==============================
def main():
    args=get_args()
    if args.single_server:
        args.judge_url,args.judge_model_name=args.gen_url,args.gen_model_name

    raw_probs={c:getattr(args,"prob_"+c) for c in CATS}
    probs=normalize_probs(raw_probs)
    cat_bins=[];acc=0.0
    for c in CATS: acc+=probs[c];cat_bins.append((c,acc))

    def draw_cat():
        r = random.random()
        for c, u in cat_bins:
            if r <= u:
                return c
        return CATS[-1]

    # Example mode
    if args.example:
        target_turns=random.choice([3,5,7]) if args.long else 3
        category=draw_cat(); bot_name=build_bot_name(category)
        seed="seed"
        persona_header=""; persona_spoken=args.persona or generate_persona(args.gen_url,args.gen_model_name,bot_name,category,seed)
        if category!="fantasy" and not args.persona and random.random()<0.5:
            persona_spoken+=" "+generate_opener(args.gen_url,args.gen_model_name,bot_name,category,persona_spoken)

        # === build 3 iterative candidates, judge best
        blocks = generate_block(args.gen_url,args.gen_model_name,bot_name,persona_spoken,persona_spoken,target_turns)
        winner,judged_final,rejected=judge_best_block(seed,blocks,args.judge_url,args.judge_model_name,bot_name)
        if winner is None: print("[WARN] Example skipped");return

        # split body vs final bot line from chosen block (no duplication in SFT)
        chosen_body, final_line_raw = split_block_and_last_bot(blocks[winner], bot_name)
        if not final_line_raw:
            print("[WARN] No final bot line found in block; using judge-selected final.")
            final_line_raw = judged_final
        final_line = ensure_final_range(args.gen_url,args.gen_model_name,bot_name,persona_spoken,final_line_raw,35,65)

        sft=format_pygmalion(chosen_body,final_line,bot_name,persona_header,persona_spoken)
        print("\n=== Example SFT ===");print(json.dumps(sft,indent=2,ensure_ascii=False))

        if not getattr(args, "sft_only", False):
            dpo={"prompt":chosen_body,"chosen":final_line,"rejected":[clamp_words(r) for r in rejected]}
            print("\n=== Example DPO ===");print(json.dumps(dpo,indent=2,ensure_ascii=False))
        return

    # Normal mode
    if not args.out: raise ValueError("--out required")
    total_resp=0;min_resp=1e9;max_resp=0;last_resp=0;total_turns=0;min_t=1e9;max_t=0;skipped=0;start=time.time()
    with open(f"{args.out}_sft.jsonl","w") as sft_out,open(f"{args.out}_dpo.jsonl","w") as dpo_out:
        for idx in range(1,args.n_prompts+1):
            target_turns=random.choice([3,5,7]) if args.long else 3
            category=draw_cat(); bot_name=build_bot_name(category)
            seed="seed"; persona_header=""; persona_spoken=args.persona or generate_persona(args.gen_url,args.gen_model_name,bot_name,category,seed)
            if category!="fantasy" and not args.persona and random.random()<0.5:
                persona_spoken+=" "+generate_opener(args.gen_url,args.gen_model_name,bot_name,category,persona_spoken)

            # 3 iterative candidates + judge
            blocks=generate_block(args.gen_url,args.gen_model_name,bot_name,persona_spoken,persona_spoken,target_turns)
            winner,judged_final,rejected=judge_best_block(seed,blocks,args.judge_url,args.judge_model_name,bot_name)
            if winner is None: skipped+=1;continue

            chosen_body, final_line_raw = split_block_and_last_bot(blocks[winner], bot_name)
            if not final_line_raw:
                print("[WARN] No final bot line found in block; using judge-selected final.")
                final_line_raw = judged_final
            final_line = ensure_final_range(args.gen_url,args.gen_model_name,bot_name,persona_spoken,final_line_raw,35,65)

            formatted=format_pygmalion(chosen_body,final_line,bot_name,persona_header,persona_spoken)
            sft_out.write(json.dumps(formatted)+"\n")

            if not getattr(args, "sft_only", False):
                for rej in rejected:
                    dpo_out.write(json.dumps({"prompt":chosen_body,"chosen":final_line,"rejected":clamp_words(rej)})+"\n")

            turns=formatted["prompt"].count("User:")+formatted["prompt"].count(f"{bot_name}:")
            total_turns+=turns;min_t=min(min_t,turns);max_t=max(max_t,turns)
            last_resp=count_words(final_line);total_resp+=last_resp;min_resp=min(min_resp,last_resp);max_resp=max(max_resp,last_resp)

            if idx%10==0 or idx==args.n_prompts:
                kept=idx-skipped;elapsed=time.time()-start;avg=elapsed/max(1,kept);eta=(args.n_prompts-idx)*avg/60
                print(f"[{idx}/{args.n_prompts}] kept={kept},skipped={skipped} {avg:.2f}s/ex, ETA {eta:.1f}m")
                print(f"  Avg final: {total_resp/max(1,kept):.1f}w (min={min_resp},max={max_resp},last={last_resp})")
                print(f"  Avg turns: {total_turns/max(1,kept):.2f} (min={min_t},max={max_t})")
                print("  === Sample SFT ===")
                print(json.dumps(formatted, indent=2, ensure_ascii=False))
                if not getattr(args, "sft_only", False):
                    print("  === Sample DPO ===")
                    if rejected:
                        print(json.dumps({
                            "prompt": chosen_body,
                            "chosen": final_line,
                            "rejected": clamp_words(rejected[0])
                        }, indent=2, ensure_ascii=False))

    print("Done.")

if __name__=="__main__": main()
