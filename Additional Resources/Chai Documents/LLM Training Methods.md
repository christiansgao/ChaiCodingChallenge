1. RLHF
   - Three stages: SFT, Reward Model Training, Reinforcement Learning (PPO / Policy Optimization)
   - How RLHF Works in Practice Model generates multiple candidate responses. Reward model scores them based on human-aligned criteria. Policy gradients (PPO) update the model to increase probability of high-reward responses.
Repeat iteratively until the model consistently prefers responses humans like.
2. SFT
   - most commonly stands for Supervised Fine-Tuning in the context of machine learning and large language models (LLMs). It is a process that adapts a pre-trained, general-purpose model to perform a specific task more accurately by training it on a smaller, high-quality, labeled dataset
3. Prompt Engineering:
   - LLMs are zero-shot or few-shot learners: they rely on the prompt to infer the task. 
   - Good prompts can:
     - Increase accuracy. 
     - Reduce ambiguity. 
     - Encourage creativity or conciseness. 
     - Make outputs safer or aligned with user intent.
4. Rejection Sampling
   -In the context of LLMs, rejection sampling is often used for preference-based sampling, e.g., during RLHF or DPO-style fine-tuning:
      - You generate multiple candidate outputs from a model.
      - You have a preference model or reward function r(y) scoring outputs.
      - You reject low-scoring outputs and only keep ones with higher reward.
      - Improves alignment with human preferences.
      - Can be seen as a sampling-time policy correction: you bias the distribution toward high-quality outputs.
5. LLM routing
   - Model to Determine which LLM to handle answering the question
6. DPO
   - Direct Preference Optimization (DPO) skips the reward model and reinforcement learning. 
   - It directly optimizes the LLM on pairs of responses labeled as “preferred” vs. “dispreferred.” 
   - The training objective comes from comparing the log-likelihoods of the preferred and dispreferred responses under the model. 
   - In essence, the model is fine-tuned so that it assigns higher probability to preferred answers.
   - Basically Logistic Loss
7. PPO (Proximal Policy Optimization)
   - What it is: A reinforcement learning (RL) algorithm, very popular because it’s relatively stable.
   - Why used in LLMs: In RLHF (Reinforcement Learning with Human Feedback), PPO is the algorithm used to fine-tune the policy (LLM) against a learned reward model. 
8. LoRA (Low-Rank Adaptation) and QLoRA (Quantized LoRA) are both parameter-efficient fine-tuning (PEFT) techniques for large language models (LLMs), but QLoRA builds upon LoRA by adding quantization to further reduce memory requirements
    - Parameters
      - r - Rank (how deep of matrix to train before multiplying an aplpying to full layer) (8 or 16)
      - How many layers to fine tune?
      - Use quantization or no
      - a - alpha scaling factor that is applied to weight changes. scaling factor is alpha / rank (.25x or 2x)
      - dropout (10%)
9. Quantization
   - Large language models (LLMs) are usually trained in full precision (FP32 or BF16), which means every weight, activation, and gradient is stored as 32-bit or 16-bit floating-point numbers. 
   - Quantization is the process of converting these values into lower-precision representations (e.g., INT8, INT4) while trying to preserve model accuracy.
   - PTQ (Post-Training Quantization)
   - QAT (Quantization-Aware Training)