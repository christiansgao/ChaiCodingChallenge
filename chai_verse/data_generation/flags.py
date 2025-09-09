# flags.py
import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # I/O
    parser.add_argument("--out", type=str,
        help="Output prefix -> writes <out>_sft.jsonl and <out>_dpo.jsonl")
    parser.add_argument("--n_prompts", type=int, default=3000,
        help="Number of conversations to generate")
    parser.add_argument("--example", action="store_true",
        help="Generate ONE example and pretty-print instead of writing files")

    # Persona & seed
    parser.add_argument("--persona", type=str, default="",
        help="Explicit persona text. If provided: goes in header AND first bot line. "
             "If not: header blank, persona spoken as first line.")
    parser.add_argument("--use_seed_chat", action="store_true",
        help="If set, use UltraChat/user prompt as seed; else just bot name/persona.")

    # Model endpoints
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

    # Generation style
    parser.add_argument("--long", action="store_true",
        help="If set, generate multi-turn convos with random length 3/5/7. "
             "Else always 3 turns.")
    parser.add_argument("--fill", action="store_true",
        help="(Currently placeholder) Intended for extension logic. "
             "When enabled, extend convo to target turns rather than regenerate.")

    # Category probabilities
    parser.add_argument("--prob_relationship", type=float, default=0.2125,
        help="Probability weight for relationship bots")
    parser.add_argument("--prob_roleplay", type=float, default=0.2125,
        help="Probability weight for roleplay bots")
    parser.add_argument("--prob_roleplay_relationship", type=float, default=0.2125,
        help="Probability weight for roleplay+relationship bots")
    parser.add_argument("--prob_fantasy", type=float, default=0.2125,
        help="Probability weight for fantasy scenarios")
    parser.add_argument("--prob_mafia", type=float, default=0.05,
        help="Probability weight for mafia archetypes")
    parser.add_argument("--prob_mafia_relationship", type=float, default=0.05,
        help="Probability weight for mafia+relationship")
    parser.add_argument("--prob_mafia_celebrity", type=float, default=0.05,
        help="Probability weight for mafia+celebrity")

    return parser.parse_args()
