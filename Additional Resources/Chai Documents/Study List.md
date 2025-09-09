1. LLM Serving Architecture
   - Model Hosting / Serving Layer
   - Inference Engine
   - Serving Infra
   - Optimization Techniques
   - Client Application Layer
   - Chai Architecture:
     - Custom Kubernetes cluster or inference engines
2. Chai Papers Areas of Study
   - Scaling
     - It is well-known that scaling up LLMs improves their performance. There are many dimensions to this scale. For example: parameters, dataset size, inference compute, context length, and the number of LLMs served. This creates an engineering challenge. 
     - Firstly, scaling up tends to increase costs; this drives us to experiment with techniques such as quantization, custom CUDA kernels, Flash Attention and KV-Caching. Secondly, at a certain scale, out-of-the-box solutions tend to breakdown. This drives us to build our own custom implementations such as our own self-managed Kubernetes cluster or inference engines.
   - Growth
     - Perplexity and Character AI took on significant investment in 2023 to amplify their organic growth. This has led to great outcomes for investors in 2024.
   - In-House Research
     - Blending Responses for User Engagment: https://arxiv.org/pdf/2401.02994
       - Blending 3 models together
3. Training Models
   - RLHF and Retention (https://blog.chai-research.com/posts/making-a-good-llm-chai-rlhf-development-cycle)
   - SFT
   - Prompt Engineering
   - Rejection Sampling
   - LLM routing
   - DPO
   - LoRA
   - Quantization
4. Problems at Chai
5. Questions for Chai
6. Inferencing Topics
   - Custom CUDA kernels
   - Flash Attention
   - KV-caching
7. Engineering
   - Microservices, Flask
   - Get, Post request ect.