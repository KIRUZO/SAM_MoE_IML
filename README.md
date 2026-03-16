---

# SAM-MoE-Forgery: Segment Anything with MoE and Prototypical Memory for Image Forensics

This repository implements a specialized framework for **Image Manipulation Localization (IML)**. It enhances the **Segment Anything Model (SAM)** by integrating a **DeepSeekMoE-inspired architecture**, a **Prototypical Memory Bank (RAG-based)**, and **Learnable CLIP Semantic Prompts**.

---

## 1. Methodology

The core philosophy of this project is to transform SAM from a semantic segmenter into a forensic segmenter by injecting multi-modal knowledge and historical "forgery experience."

### 1.1 DeepSeekMoE-inspired Adapters
Instead of standard monolithic adapters, we inject a **DeepSeekMoE** structure into the SAM Image Encoder's Transformer blocks. This allows the model to handle the diversity of forgery types (Splicing, Copy-Move, etc.).

*   **Shared Experts:** Capture universal forensic traces (e.g., JPEG compression artifacts).
*   **Routed Experts:** Fine-grained experts that are sparsely activated via a **Top-K Router** to specialize in specific manipulation signatures.

**Mathematical Formula:**
For a hidden state $x$, the output of the MoE layer $y$ is:
$y = \text{Base}(x) + \sum_{j=1}^{N_s} E_{shared,j}(x) + \sum_{k \in \text{TopK}} \mathcal{G}(x)_k \cdot E_{routed,k}(x)$
Where:
- $\text{Base}(x)$ is the frozen SAM linear output.
- $N_s$ is the number of shared experts.
- $\mathcal{G}(x)$ is the Softmax-gated router output for the routed experts.

### 1.2 Prototypical Memory Bank (RAG for Forensics)
To provide the model with "prior knowledge" of what forgery looks like, we implement a **Prototypical Memory Bank** containing 16 forgery prototypes and 16 authentic prototypes.

*   **Retrieval (Guidance):** The model compares current image features against the memory bank to generate a **Similarity Guidance Map**.
    $$\text{Guidance} = \max(\text{Sim}(x, P_{forgery})) - \max(\text{Sim}(x, P_{authentic}))$$
*   **Update (Slot-based Momentum):** During training, the model identifies the "nearest" expert slot in the memory bank for the current batch's features and updates it using momentum:
    $$P_{target} = m \cdot P_{target} + (1-m) \cdot \text{BatchCenter}$$
    This ensures each prototype slot specializes in a specific forensic pattern (e.g., one for noise, one for edge blurring).

### 1.3 Semantic CLIP Mapper (Multimodal Prompts)
Instead of using fixed points or boxes, we use **Learnable Context Optimization (CoOp)**.
1.  **Learnable Context:** We define a set of learnable tokens $[V]_1, [V]_2, \dots, [V]_n$ combined with CLIP's SOS/EOS tokens.
2.  **Cross-Attention:** A `SemanticCLIPMapper` uses learnable queries to extract forensic-relevant semantics from the CLIP text embedding.
3.  **Injection:** These are fed into the SAM Mask Decoder as **Sparse Prompt Embeddings**.

---

## 2. Why it Works?

1.  **Expert Specialization:** The MoE router prevents interference between different forensic tasks. For example, the expert trained on "copy-move" patterns won't be diluted by data from "inpainting" tasks.
2.  **Explicit Forensic Bias:** By adding the `forgery_guidance` from the memory bank directly to the `dense_prompt_embeddings`, we force the SAM Decoder to pay attention to regions that historically "look like" forgeries.
3.  **Semantic Guidance:** Using CLIP features as prompts allows the model to leverage high-level linguistic concepts of "objects" vs "manipulations," which regular SAM lacks.
4.  **Stability:** By freezing the backbone and using low-rank ($r=16$) experts, the model maintains SAM's spatial reasoning while becoming sensitive to forensic noise.

---

## 3. Loss Function

The model is optimized using a balanced **Forgery Loss** without requiring complex load-balancing weights, focusing on high-quality mask generation:

$$\mathcal{L}_{total} = \mathcal{L}_{BCE}(\text{pred}, \text{gt}) + \mathcal{L}_{Dice}(\text{pred}, \text{gt})$$

- **BCE:** Standard pixel-wise classification.
- **Dice:** Handles class imbalance (when the tampered area is very small compared to the whole image).

---

## 4. Usage

### Installation
```bash
pip install -r requirements.txt
# Requires: torch, segment_anything, clip, opencv-python, tqdm
```

### Training
The script supports **Mixed Precision (AMP)** for faster training on modern GPUs.

```bash
python train.py \
    --train_root /path/to/dataset \
    --sam_ckpt ./weights/sam_vit_h_4b8939.pth \
    --output_dir ./runs/my_experiment \
    --batch_size 1 \
    --lr 1e-5 \
    --gpu 0 \
    --mixed_precision
```

### Data Structure
The dataset should follow this convention:
- `Tp/`: Manipulated images (Target images).
- `Gt/`: Binary ground truth masks.

---

## 5. Implementation Details

| Component | Specification |
| :--- | :--- |
| **Backbone** | SAM ViT-H (Frozen) |
| **MoE Config** | 1 Shared Expert, 6 Routed Experts, Top-3 Selection |
| **Memory Bank** | 256-dim features, 16 slots per class |
| **Input Size** | 1024x1024 (Image), 256x256 (GT Mask) |
| **Optimizer** | AdamW with $10^{-5}$ Learning Rate |

--- 

**Author:** [KIRUZO]  
**License:** MIT License
