# flags.py
import argparse
from constants import CATS  # so we can use consistent categories

def get_args():
    parser = argparse.ArgumentParser()

    # ==============================
    # I/O
    # ==============================
    parser.add_argument("--out", type=str,
        help="Output prefix -> writes <out>_sft.jsonl and <out>_dpo.jsonl")
    parser.add_argument("--n_prompts", type=int, default=3000,
        help="Number of conversations to generate")
    parser.add_argument("--example", action="store_true",
        help="Generate ONE example and pretty-print instead of writing files")
    parser.add_argument("--out_path", type=str, default=None,
        help="Optional file path to append raw pygmalion JSONL")
    parser.add_argument("--print_json", action="store_true",
        help="Print only pygmalion JSON (for piping)")

    # ==============================
    # Persona & seed
    # ==============================
    parser.add_argument("--persona", type=str, default="",
        help="Explicit persona text. If provided: goes in header AND first bot line. "
             "If not: header blank, persona spoken as first line.")
    parser.add_argument("--use_seed_chat", action="store_true",
        help="If set, use UltraChat/user prompt as seed; else just bot name/persona.")
    parser.add_argument("--seed", type=str, default="",
        help="Conversation seed text")
    parser.add_argument("--ultra_seed", type=int, default=None,
        help="Random seed for reproducibility")

    # ==============================
    # Model endpoints
    # ==============================
    parser.add_argument("--sft_only", action="store_true", help="If set, only generate SFT data (no DPO).")

    parser.add_argument("--gen_url", type=str, default="http://localhost:8000/v1/chat/completions",
        help="Generator endpoint URL")
    parser.add_argument("--gen_model_name", type=str, default="gen7b",
        help="Name of generator model as served by vLLM/OpenAI")
    parser.add_argument("--judge_url", type=str, default="http://localhost:8001/v1/chat/completions",
        help="Judge endpoint URL")
    parser.add_argument("--judge_model_name", type=str, default="judge7b",
        help="Name of judge model as served by vLLM/OpenAI")
    parser.add_argument("--single_server", action="store_true",
        help="If set, reuse generator server/model as judge.")

    # ==============================
    # Generation style
    # ==============================
    parser.add_argument("--long", action="store_true",
        help="If set, generate multi-turn convos with random length 3/5/7. Else always 3 turns.")
    parser.add_argument("--fill", action="store_true",
        help="(Placeholder) Extend convo to target turns instead of regenerating.")
    parser.add_argument("--turns", type=int, default=8,
        help="Total lines (bot+user), must end with bot")
    parser.add_argument("--n_drafts", type=int, default=3,
        help="How many conversation drafts to generate")
    parser.add_argument("--no_judge", action="store_true",
        help="Skip judging; take first draft")
    parser.add_argument("--keep_all_drafts", action="store_true",
        help="If set, include all drafts in JSON output")

    # ==============================
    # Category probabilities
    # ==============================
    parser.add_argument("--prob_relationship", type=float, default=0.18,
        help="Probability weight for relationship bots")
    parser.add_argument("--prob_roleplay", type=float, default=0.18,
        help="Probability weight for roleplay bots")
    parser.add_argument("--prob_roleplay_relationship", type=float, default=0.18,
        help="Probability weight for roleplay+relationship bots")
    parser.add_argument("--prob_fantasy", type=float, default=0.18,
        help="Probability weight for fantasy scenarios")
    parser.add_argument("--prob_celebrity", type=float, default=0.18,
        help="Probability weight for celebrity bots")
    parser.add_argument("--prob_mafia", type=float, default=0.05,
        help="Probability weight for mafia archetypes")
    parser.add_argument("--prob_mafia_relationship", type=float, default=0.05,
        help="Probability weight for mafia+relationship")
    parser.add_argument("--prob_mafia_celebrity", type=float, default=0.05,
        help="Probability weight for mafia+celebrity")

    # ==============================
    # Hyperparameters
    # ==============================
    parser.add_argument("--gen_temperature", type=float, default=0.9)
    parser.add_argument("--gen_top_p", type=float, default=0.95)
    parser.add_argument("--judge_temperature", type=float, default=0.0)
    parser.add_argument("--judge_top_p", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=300,
        help="HTTP timeout seconds for model calls")

    # token budgets
    parser.add_argument("--persona_max_tokens", type=int, default=180)
    parser.add_argument("--opener_max_tokens", type=int, default=40)
    parser.add_argument("--block_max_tokens", type=int, default=640)
    parser.add_argument("--tighten_max_tokens", type=int, default=160)

    # ==============================
    # Closing line constraints
    # ==============================
    parser.add_argument("--final_min_words", type=int, default=35)
    parser.add_argument("--final_max_words", type=int, default=65)

    # ==============================
    # Overrides
    # ==============================
    parser.add_argument("--bot_name", type=str, default="")
    parser.add_argument("--persona_override", type=str, default="")
    parser.add_argument("--opener_override", type=str, default="")
    parser.add_argument("--celebrity_gender", choices=["any","female","male"], default="any",
        help="Bias celebrity selection")

    return parser.parse_args()
