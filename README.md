# I Can(’t) Help With That  
### Finding Refusal and Truthful Directions in LLMs

**Authors:** Yilin Huang, Jiyuan Ji, Megan Li, Sarah Stromswold  

This repository contains code for our paper:

> *I Can(’t) Help With That: Finding Refusal and Truthful Directions in LLMs*

We investigate whether two important LLM safety behaviors — **refusal** and **truthfulness** — can be captured and manipulated using a single activation direction in model representation space.

---

## Overview

Large language models must:

- **Refuse** harmful or unsafe requests  
- **Be truthful**, meaning they should admit when a question is unanswerable instead of hallucinating  

Prior work (Arditi et al., 2024) showed refusal may be mediated by a single direction in activation space.  

We extend this by:
1. Testing refusal directions across multiple model families and sizes  
2. Attempting to extract a similar direction for truthfulness  

---

## Key Findings

### Refusal
- A **single direction** can strongly control refusal behavior  
- Adding the direction induces near-universal refusal  
- Removing it significantly reduces refusal  
- Larger models are more robust to jailbreak attempts  

### Truthfulness
- No single linear direction reliably captures truthfulness  
- Interventions either had little effect or harmed performance  
- Truthfulness appears more complex and distributed than refusal  

---

## Models Tested

- Yi-6B-Chat  
- LLaMA-2 (7B, 13B)  
- LLaMA-3-8B-Instruct  
- Gemma-2B  
- Qwen-1.8B-Chat  

Truthfulness experiments were conducted on:
- Yi-6B-Chat  
- LLaMA-2-7B  

---

## Datasets

**Refusal**
- AdvBench  
- MaliciousInstruct  
- HarmBench  
- Alpaca (harmless prompts)

**Truthfulness**
- SQuAD 2.0  

---

## Method

1. Compute mean activations for two behavior classes  
2. Take the difference-in-means vector  
3. Select the best-performing layer/token  
4. Apply:
   - **Activation Addition** (steering)
   - **Directional Ablation**

Evaluation uses refusal substring matching and LLM-as-a-judge classifiers.

---

## Setup

```bash
pip install -r requirements.txt

Download the required HuggingFace model checkpoints before running experiments.

---

## Example Commands

Extract refusal direction:

```bash
python extract_refusal_direction.py --model llama2-7b
```

Extract truthfulness direction:

```bash
python extract_truthfulness_direction.py --model yi-6b
```

Run interventions:

```bash
python run_intervention.py --mode addition
python run_intervention.py --mode ablation
```

---

## Main Result Summary

| Behavior      | Single Direction Works? |
|--------------|------------------------|
| Refusal      | Yes                    |
| Truthfulness | No                     |

Refusal appears relatively linear and controllable.  
Truthfulness appears structurally more complex.

---

## Citation

```bibtex
@article{huang2025refusaltruthfulness,
  title={I Can(’t) Help With That: Finding Refusal and Truthful Directions in LLMs},
  author={Huang, Yilin and Ji, Jiyuan and Li, Megan and Stromswold, Sarah},
  year={2025}
}
```
