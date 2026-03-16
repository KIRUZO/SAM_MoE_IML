---

# SAM-MoE-Forgery: Segment Anything with MoE and Prototypical Memory for Image Forensics

This repository implements a specialized framework for **Image Manipulation Localization (IML)**. It enhances the **Segment Anything Model (SAM)** by integrating a **DeepSeekMoE-inspired architecture**, a **Prototypical Memory Bank (RAG-based)**, and **Learnable CLIP Semantic Prompts**.

---

## 1. Methodology

The core philosophy of this project is to transform SAM from a semantic segmenter into a forensic segmenter by injecting multi-modal knowledge and historical "forgery experience."

### 1.1 DeepSeekMoE-inspired Adapters
Instead of standard monolithic adapters, we inject a **DeepSeekMoE** structure into the SAM Image Encoder's Transformer blocks. This allows the model to handle the diversity of forgery types.

*   **Shared Experts:** Capture universal forensic traces (e.g., JPEG compression artifacts).
*   **Routed Experts:** Fine-grained experts activated via a **Top-K Router** ($K=3$) to specialize in specific manipulation signatures.

**Mathematical Formula:**
For a hidden state $x$, the output of the MoE layer $y$ is:
```math
y = \text{Base}(x) + \sum_{j=1}^{N_s} E_{shared,j}(x) + \sum_{k \in \text{TopK}} \mathcal{G}(x)_k \cdot E_{routed,k}(x)
```
Where $\mathcal{G}(x)$ is the Softmax-gated router output and $E(x)$ are low-rank bottleneck adapters ($rank=16$).

### 1.2 Prototypical Memory Bank (Forensic RAG)
The `PrototypicalMemoryBank` acts as a Retrieval-Augmented Generation (RAG) component. It maintains two sets of learnable buffers: **Forgery Prototypes** $\mathbf{P}_f$ and **Authentic Prototypes** $\mathbf{P}_a$, each containing 16 specialized slots.

#### A. Retrieval (Similarity Guidance)
For every visual token $x_{hw}$ in the image embedding, we calculate its "forensic evidence" by measuring cosine similarity against all stored prototypes:

1.  **Similarity Calculation:**
```math
    \mathcal{S}_f = \max_{j \in \{1 \dots 16\}} \left( \frac{x_{hw} \cdot \mathbf{P}_{f,j}}{\|x_{hw}\| \|\mathbf{P}_{f,j}\|} \right), \quad \mathcal{S}_a = \max_{j \in \{1 \dots 16\}} \left( \frac{x_{hw} \cdot \mathbf{P}_{a,j}}{\|x_{hw}\| \|\mathbf{P}_{a,j}\|} \right)
```
2.  **Guidance Map Generation:**
    The final guidance $\mathcal{G}$ is the relative difference, highlighting regions that deviate from authentic patterns:
```math
    \text{Guidance} = \mathcal{S}_f - \mathcal{S}_a
```
This guidance map is added to SAM's **dense prompt embeddings**, forcing the decoder to focus on suspicious regions.

#### B. Slot-based Momentum Update
During training, the memory bank is updated dynamically. Instead of updating all prototypes, we use a **competitive routing** strategy:
1.  Calculate the mean feature $\mu_{batch}$ of all forgery pixels in the current batch.
2.  Find the "nearest" prototype slot: $j^* = \arg\max_j (\text{Sim}(\mu_{batch}, \mathbf{P}_{f,j}))$.
3.  Apply momentum update to **only** that specific slot:
```math
    \mathbf{P}_{f,j^*} \leftarrow m \cdot \mathbf{P}_{f,j^*} + (1-m) \cdot \mu_{batch}
```
*(where $m=0.9$)*. This ensures each slot specializes in a distinct forensic artifact (e.g., noise, resampling, or blur).

### 1.3 Semantic CLIP Mapper (Multimodal Prompts)
To guide SAM with high-level semantic concepts, we utilize a frozen CLIP model with a **Learnable Prompt** mechanism.

#### A. Learnable Context Construction
We do not use hand-crafted text prompts. Instead, we optimize a continuous **Learnable Context** $\mathbf{V} \in \mathbb{R}^{16 \times 512}$:
```math
\text{Prompt} = [\text{SOS}] \oplus [\mathbf{V}]_1 \oplus [\mathbf{V}]_2 \oplus \dots \oplus [\mathbf{V}]_{16} \oplus [\text{EOS}]
```
This sequence is passed through the frozen CLIP Transformer. The feature $\mathbf{z}_{text}$ is extracted from the `[EOS]` token position.

#### B. Semantic Mapping via Cross-Attention
The `SemanticCLIPMapper` converts the 512-dim CLIP feature into 4 discrete SAM sparse tokens ($256$-dim) using Cross-Attention:
1.  **Query:** 4 learnable query tokens $\mathbf{Q} \in \mathbb{R}^{4 \times 256}$.
2.  **Key/Value:** The transformed CLIP text embedding $\mathbf{z}_{text}$.
```math
\text{Sparse Prompts} = \text{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d}}\right) \mathbf{V}
```
These 4 tokens are injected into the **SAM Mask Decoder** as sparse prompt embeddings, providing semantic guidance for the localization task.

---

## 2. Why it Works?

1.  **Expert Specialization:** The MoE router prevents interference between different forensic tasks. An expert specialized in "Copy-Move" patterns won't be diluted by "Inpainting" data.
2.  **Historical Awareness:** The Memory Bank allows the model to "remember" forensic signatures across batches. If the current image contains artifacts similar to previously seen forgeries, the Guidance Map will strongly signal the Decoder.
3.  **Language-Vision Synergy:** The Learnable CLIP prompts allow the model to learn a forensic-specific "vocabulary" that maps text-space manipulation concepts to pixel-space boundaries.
4.  **Parameter Efficiency:** By freezing the SAM backbone and CLIP, we only train the MoE layers, the Mapper, and the Memory slots, making the model highly robust to overfitting on small forensic datasets.

---

## 3. Loss Function

The model is optimized using a combined **Forgery Loss**:

```math
\mathcal{L}_{total} = \mathcal{L}_{BCE}(P, Y) + \mathcal{L}_{Dice}(P, Y)
```

- **BCE (Binary Cross-Entropy):** Ensures pixel-wise classification accuracy.
- **Dice Loss:** Addresses the heavy class imbalance typical in IML (where the tampered region is usually much smaller than the background).

---

## 4. Usage

### Installation
```bash
pip install -r requirements.txt
# Main dependencies: torch, segment_anything, clip, opencv-python
```

### Training
The script supports **Mixed Precision (AMP)** for efficient training.

```bash
python train.py \
    --train_root /path/to/dataset \
    --sam_ckpt ./weights/sam_vit_h_4b8939.pth \
    --output_dir ./runs/final_moe \
    --batch_size 1 \
    --lr 1e-5 \
    --gpu 0 \
    --mixed_precision
```

### Data Structure
- `Tp/`: Manipulated images.
- `Gt/`: Binary ground truth masks ($255$ for tampered, $0$ for authentic).

---

## 5. Implementation Details

| Component | Specification |
| :--- | :--- |
| **Backbone** | SAM ViT-H (Frozen) |
| **MoE Configuration** | 1 Shared Expert, 6 Routed Experts, Top-3 Selection |
| **Memory Bank** | 256-dim features, 16 slots per class, Momentum $0.9$ |
| **Prompt Tokens** | 16 Learnable Context tokens + 4 Sparse SAM tokens |
| **Input Size** | 1024x1024 (Image) $\rightarrow$ 64x64 (Internal Embedding) |

--- 

**Author:** [Zireal]  
**License:** MIT License
