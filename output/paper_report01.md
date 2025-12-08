# 复现规划：DeepSeek-R1: Large-scale Reinforcement Learning to Emerge Reasoning + Cold-start SFT and Distillation

**任务类型：** Emergent reasoning via post-training (RL) + distillation for reasoning-intensive tasks (math, code, logic)
**模型家族：** Transformer decoder LLMs with RL-based post-training (GRPO) and SFT distillation to dense models (Qwen / Llama family)

## 主要数据集
- AIME 2024 (AIME) - math contest benchmark
- MATH-500 - math reasoning benchmark
- MMLU / MMLU-Pro - multitask knowledge benchmarks
- GPQA Diamond / SimpleQA - question answering / factual benchmarks
- LiveCodeBench / HumanEval-Mul / Codeforces - code reasoning and competition problems
- SWE-Bench Verified - software engineering evaluation
- C-Eval / CMMLU / CLUEWSC - Chinese/ multilingual benchmarks
- Internal cold-start CoT data (~thousands) and generated RL trajectory data (reported ~600k reasoning + ~200k non-reasoning ≈ 800k samples) (paper-provided)

## 关键参考
- DeepSeek (prior): DeepSeek-V3 / Deepseekmath (Shao et al., 2024) — base model / prior work referenced in paper (explicit)
- GRPO: Group Relative Policy Optimization (Shao et al., 2024) — RL method used (explicit)
- OpenAI o1 series / 'Learning to Reason with LLMs' (OpenAI, 2024) — benchmark comparator and inspiration for long CoT (explicit)
- Lightman et al., 2023; Uesato et al., 2022; Wang et al., 2023 — process reward / CoT related prior work (explicit)
- Qwen2.5 and QwQ-32B-Preview (Qwen project) — distillation targets and baselines (explicit)
- Llama 3 (AI@Meta / Dubey et al., 2024) — distillation targets (explicit)
- Reward hacking / RM risks: Gao et al., 2022 — motivation to avoid neural RM early (explicit)
- MCTS / AlphaZero style search references (Silver et al., 2017) — attempted but limited (explicit)
- trl / trlx (community libraries) — practical libraries to implement RLHF/actor training (inferred/tool suggestion)

## 复现步骤（高层）
1. (Repro level 0: small-scale distillation baseline) Collect/generate an 800k-sample SFT dataset similar to paper: ~600k reasoning (CoT) samples + ~200k non-reasoning samples. Start from an open Qwen/Llama checkpoint and fine-tune with supervised learning on this dataset. Evaluate on small subsets of AIME / MATH / Codeforces to confirm gains.
2. (Implement reward functions) Implement rule-based reward modules: (1) accuracy rewards — automated checks for math (answer matching), code (unit tests / compilation + testcases), (2) formatting reward — enforce CoT enclosed in special tokens, (3) language-consistency reward — ratio of tokens in target language. Keep rewards additive and normalized.
3. (DeepSeek-R1-Zero: pure RL experiment) Starting from base pre-trained checkpoint (no SFT warm-start), implement GRPO to perform policy updates. For each prompt sample G outputs from the current policy, compute per-output rewards, advantages (group-normalized), and apply PPO-like clipped objective plus a KL penalty to a reference policy; use group-based baseline as in GRPO.
4. (Cold-start + multi-stage training for DeepSeek-R1) Create a small cold-start CoT SFT set (thousands) and fine-tune base model. Then run reasoning-oriented RL (GRPO) to improve CoT and accuracy. Near RL convergence, run rejection sampling on RL checkpoints to collect high-quality trajectories, combine with other SFT data (writing, QA, translation) to build the ~800k SFT dataset and re-fine-tune model. Then run a final RL phase (all-scenarios RL) with combined reward signals and safety/unharmfulness objectives.
5. (Rejection sampling & SFT loop) From RL checkpoints, sample multiple responses per prompt, keep only high-reward / human-readable responses (filter language-mixing, long raw code blocks, unreadable outputs). Use these to create teacher SFT data for distillation.
6. (Distillation to smaller dense models) Use the collected 800k SFT dataset to SFT-fine-tune multiple dense models (1.5B, 7B, 14B, 32B, 70B) — no RL on the small models initially. Evaluate on target benchmarks and compare to larger-model RL runs.
7. (Evaluation and metrics) For pass@1 style metrics: sample N responses per prompt (paper used 64 for many pass@1 estimates), compute pass@1. Use generation settings (temp=0.6, top-p=0.95). For other tasks use dataset-specific evaluation protocols (MMLU zero-shot prompts, LiveCodeBench CoT format, Codeforces rating estimation).
8. (Ablations) Ablate: (a) GRPO vs PPO baseline; (b) with/without language-consistency reward; (c) pure RL vs cold-start SFT+RL; (d) distill-from-big vs RL-small.

## 关键超参数/设置（需要重点关注）
- Generation / evaluation settings (explicit in paper): max generation length = 32,768 tokens (paper); temperature = 0.6; top-p = 0.95; use 64 samples per prompt to estimate pass@1 (explicit).
- GRPO objective extras: clipping epsilon and KL penalty beta are used in the GRPO objective (paper formula) — specific numeric epsilon/beta not provided in paper (uncertain).
- Group size G for GRPO sampling: paper describes sampling a group {o1..oG} from old policy; paper used G sufficient to compute stable group baseline — exact value not stated (uncertain).
- RL training budget: DeepSeek-R1-Zero trained for 'thousands' of RL steps; a reported Qwen-32B RL run ran >10k steps (explicit/partial). Expect to need thousands–tens of thousands steps to see strong gains (inferred).
- Rejection-sampling & SFT dataset sizes (explicit): ~600k reasoning samples collected via rejection-sampling, ~200k non-reasoning samples, total ≈ 800k SFT samples.
- Distillation fine-tuning: typical SFT hyperparams to try — learning rate 1e-5 ~ 5e-5 (adaptive), batch size large enough to process dataset (e.g., effective batch 512–2048 tokens), sequence length adequate for CoT (>=2048, paper used up to 32k). These exact values are not reported (inferred).
- Evaluation sampling for few-shot/zero-shot: follow dataset guidance; for MMLU-Redux use Zero-Eval prompt format (paper).

## 潜在坑点
- Reward hacking: using neural reward models early can be exploited; the paper avoided neural RMs in initial large RL runs (explicit). If you add RMs, watch for reward hacking and need for RM retraining / auditing.
- Language mixing / readability: pure RL without language or formatting constraints can produce unreadable or mixed-language CoTs — implement language-consistency and formatting rewards and strict filters in rejection sampling (paper reports this issue explicitly).
- Compute cost & sample complexity: replicating large-scale RL on large models (32B+) requires massive compute; start small. Paper reports vast training (thousands to >10k RL steps) — resource heavy (explicit/inferred).
- Reproducibility of GRPO specifics: paper gives formula but omits numeric hyperparameters (epsilon, beta, group size G, learning rates, batch sizes). These will influence stability and are not fully specified (uncertain).
- Evaluation bias: many evaluations rely on model judges (GPT-4-Turbo-1106) or long CoT outputs; judge / prompt design can change leaderboard positions. Also pass@1 depends on sampling and filtering strategy.
- Filtering quality: the rejection-sampling pipeline that created the 600k reasoning samples must aggressively filter mixed/low-quality outputs; insufficient filtering will degrade SFT distillation results.
- Scaling distillation vs RL: paper finds distillation from a stronger teacher outperforms running RL directly on smaller models — naive RL on small models may underperform unless you invest heavy compute.
- Safety / refusal behavior: safety RL phases can raise refusal rates and reduce factual recall on some benchmarks (paper notes SimpleQA drop when safety RL applied).

## 不确定/需要人工判断的点
- Explicitly stated in paper: use of GRPO, two-stage RL + SFT pipeline for DeepSeek-R1, dataset composition (~600k reasoning + ~200k non-reasoning = ~800k), generation settings (max tokens 32,768; temp=0.6; top-p=0.95), sampling 64 responses for pass@1 in many experiments (explicit).
- Unclear / inferred from paper or omitted: exact GRPO hyperparameters (epsilon clipping value, beta KL weight), group size G used to compute advantages, actor learning rate, optimizer, batch size, gradient-accumulation schedule — paper does not provide numeric values (uncertain).
- Training compute & wall-clock: paper does not publish exact GPU/TPU counts, FLOPs or total RL S tokens; they mention 'thousands' or >10k steps in one setting — exact compute requirement is uncertain (inferred heavy).
- Rejection-sampling thresholds and filtering heuristics: paper describes filtering mixed-language and unreadable outputs but does not give deterministic rules/thresholds — implement conservative filters and validate with human inspection (uncertain).
- Exact seed / evaluation protocols for pass@1 vs cons@64 etc.: paper reports numbers but replicability may vary with prompt templates and judge seeds (uncertain).
- Open-source availability: paper claims they open-sourced checkpoints and distilled models; availability, licenses and exact artifact locations may vary (uncertain — check project repo).

## 建议的起始实现
Practical reproducibility path (low-to-medium compute): 1) Start with the distillation experiment — easiest and most compute-efficient way to reproduce gains: pick an open checkpoint (Qwen2.5-7B or Llama-8B) and fine-tune with the provided/constructed SFT dataset (approx 800k samples: ~600k reasoning CoT + ~200k non-reasoning). 2) Use standard SFT fine-tuning tooling (HF Trainer / DeepSpeed / LoRA if GPU-limited). Use sequence length >=2048 for CoT (paper used up to 32k but this is costly). 3) Reproduce evaluation on a small subset of AIME / MATH-500 and LiveCodeBench; use generation settings temp=0.6, top-p=0.95 and sample 16–64 outputs per prompt to estimate pass@1 (paper used 64). 4) After SFT baseline works, implement a small-scale GRPO-like RL loop on the 7B checkpoint with a limited RL budget (few thousand steps) on a curated math+code subset, using rule-based rewards (math answer checking, unit tests for code) to observe directional improvements. Libraries to bootstrap from: HuggingFace Transformers + trl / trlx or Ray RLlib for custom policy updates. 5) Iterate: add rejection-sampling + filtering to collect higher-quality SFT samples and re-run SFT distillation. 6) When/if you have access to larger compute, scale actor model size and RL steps and tune GRPO epsilon/beta to stabilize training. 7) Throughout, log rewards, token lengths, language-mix ratios, and evaluate with both automated metrics and sample human checks to avoid reward-hacking artifacts.