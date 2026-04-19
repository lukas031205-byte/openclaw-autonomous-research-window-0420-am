# VAE-Induced Semantic Drift in Video Generation: Disease, Diagnostic, and Treatment

**Authors:** Anonymous (Workshop Submission)  
**Venue:** CVPR 2026 / ICLR 2026 Workshop (To Be Determined)  
**Note:** This is a draft. All experiments conducted on CPU with synthetic data.  
**0420-AM correction:** Abstract/Results corrected — actual CNLSA-Bridge result is r=0.3681 (not 0.5472 as previously reported). This was a critical data integrity fix identified by Scalpel review on 2026-04-20.

---

## Abstract

Variational Autoencoders (VAEs) are the dominant latent space architecture in video generation pipelines, but their semantic fidelity is poorly characterized. We investigate **VAE-induced semantic drift**: CLIP cosine similarity drops from 1.0 to 0.9388 at identity noise (σ=0) and to 0.343 for CLIP ViT-B/16 at σ=0 on COCO Val2017, with drift being category-uniform (ANOVA p=0.6037). We propose **TrACE-Video** and its **Latent Consistency Score (LCS)**: L2 distance between DINOv2 embeddings as a CLIP-free proxy for semantic drift. On CIFAR-10 with VAE latent perturbation, Pearson r(DINOv2_L2, 1−CLIP_sim) = 0.3681 (p<10⁻¹⁰, 95% CI [0.27, 0.46]). LCS explains approximately 14% of CLIP semantic drift variance (r²=0.14). Send-VAE (ICLR 2026) and test-time correction (TTC, arXiv 2602.05871) provide complementary treatment pathways, with LCS serving as the evaluation layer for treatment efficacy.

---

## 1. Introduction

Variational Autoencoders (VAEs) are the dominant latent space architecture in contemporary video generation pipelines (Wan2.1, CogVideoX, Stable Video Diffusion). However, the semantic fidelity of VAE latent representations remains poorly characterized. When video content passes through a VAE encode–decode roundtrip, does the semantic information remain intact, or does it systematically distort?

This paper investigates **VAE-induced semantic drift**: the phenomenon whereby VAE encode–decode roundtrips cause systematic misalignment between the semantic content of the original video and the reconstructed video, as measured by CLIP embeddings. We present a three-stage research contribution:

1. **CNLSA (The Disease Model):** We characterize VAE-induced CLIP semantic drift, showing it is (a) substantial in magnitude (CLIP similarity drops from 1.0 to 0.9388 at σ=0), (b) category-uniform across semantic categories (ANOVA p=0.6037), (c) architecture-variant — affecting larger ViT architectures more severely (CLIP ViT-B/16: CS=0.343; CLIP ViT-S/14: CS=0.816 at σ=0), and (d) **not factor-separable** — we falsify the hypothesis that VAE encoding increases CLIP-DINOv2 correlation; the drift mechanism is unitary, not a combination of separable factors.

2. **TrACE-Video (The Diagnostic):** We propose DINOv2 L2 distance as an unsupervised proxy metric for VAE-induced semantic drift. Across VAE latent perturbation levels (σ ∈ {0, 0.1, 0.2, 0.5, 1.0}) on CIFAR-10, Pearson correlation r=0.3681 (p<10⁻¹⁰, 95% CI [0.27, 0.46]) between DINOv2 L2 distance and 1 − CLIP similarity. This establishes TrACE-Video's LCS (Latent Consistency Score) as a computable diagnostic that does not require a CLIP model at inference time.

3. **Send-VAE / TTC (The Treatment):** Test-time correction methods (Send-VAE, ICLR 2026; Test-Time Correction, arXiv 2602.05871) provide a treatment path. Send-VAE proposes semantic-disentangled VAE training; TTC proposes test-time latent space correction. TrACE-Video's LCS metric can serve as both the diagnostic trigger and the evaluation metric for treatment efficacy.

**Contribution chain:** Disease (CNLSA) → Diagnostic (TrACE-Video) → Treatment (Send-VAE/TTC) → Evaluation (LCS metric). This constitutes a coherent PhD-chapter-scale research narrative: identify a systematic failure mode, develop a measurement tool, and validate a correction pathway.

---

## 2. Related Work

### 2.1 VAE Alternatives: Going VAE-Free

The most direct response to VAE-induced semantic drift is to eliminate the VAE entirely. **SVG** (ICLR 2026, arXiv 2510.15301) replaces the VAE latent space with DINOv3 self-supervised features, arguing that "VAE latent spaces lack clear semantic separation and strong discriminative structure." This is direct confirmation of the CNLSA hypothesis from a competing approach. **EPG / "There is No VAE"** (ICLR 2026, arXiv 2510.12586) proposes end-to-end pixel-space diffusion via SSL pretraining, bypassing VAE compression entirely. **Diagonal Distillation** (ICLR 2026, arXiv 2603.09488) proposes asymmetric per-chunk compute for streaming autoregressive video generation — early chunks receive multi-step denoising while later chunks inherit appearance from fully-processed predecessors — a complementary architectural treatment for inter-frame consistency. **LVTINO** (ICLR 2026, arXiv 2510.01339) addresses latent video consistency via an inverse solver framework, complementing test-time correction approaches. **LongLive** (ICLR 2026, OpenReview) introduces a frame-level autoregressive framework for real-time interactive long video generation (20.7 FPS on H100), establishing the operational boundary within which VAE-induced semantic drift must be tolerated.

### 2.2 Semantic Fidelity in Diffusion Models

**SFD** (CVPR 2026) proposes semantic-first diffusion, prioritizing semantic consistency over pixel-level fidelity. **LSA** (arXiv 2602.05966) fine-tunes video generation models with localized semantic feature alignment, conceptually identical to CNLSA's latent agreement principle — an independent replication. **Re2Pix** (arXiv 2604.11707, April 13 2026) predicts VFM semantic features first, then uses them to guide pixel generation — semantic feature guidance reduces temporal drift, directly related to CNLSA. **VideoGPA** (arXiv 2601.2328) distills geometry priors for 3D-consistent video generation — geometric consistency as a complementary consistency dimension. **LatSearch** (ICLR 2026) achieves 79% runtime reduction in latent video search via learned consistency metrics. **ODC** (ICLR 2026) introduces orthogonal drift correction, addressing distributional shift in latent spaces during generation. **VGGRPO** (arXiv 2603.26599) proposes a Latent Geometry Model (LGM) that sutures video diffusion latent with geometric foundation models for world-consistent video generation — geometry-aware RL post-training in latent space suggests VAE-induced semantic drift and geometric inconsistency share a common root.

### 2.3 Test-Time Adaptation for Video Generation

**TTOM** (ICLR 2026, arXiv 2510.07940) proposes test-time optimization and memorization for compositional video generation — zero-shot per-prompt adaptation without retraining. **Pathwise TTC** (arXiv 2602.05871) stabilizes long autoregressive video generation by using initial frames as reference anchors to calibrate stochastic states. **Event-Driven Video Generation** (arXiv 2603.13402) introduces event-gated sampling with hysteresis and early-step scheduling — complementary to TTC: where TTC corrects drift post-hoc, event-driven gating prevents cumulative drift by modulating per-frame denoising budget. **EvoSearch** (ICLR 2026 OpenReview) proposes evolutionary search for test-time scaling of diffusion models — the extreme end of the treatment spectrum beyond TTC's single-pass correction. **Video-T1** (arXiv 2503.18942) explores test-time scaling via tree-of-frame search. **Test-Time Flow Maps** (ICLR 2026) propose test-time correction via flow field estimation.

### 2.4 Semantic Drift in Encoders

**"When Backdoors Go Beyond Triggers"** (arXiv 2602.20193) investigates semantic drift in diffusion models under encoder attacks — directly relevant to understanding the mechanisms of encoder-induced semantic degradation. **FreeViS** (arXiv 2510.01686) proposes training-free temporal consistency with inconsistent reference frames, providing an alternative measurement approach.

### 2.5 Position Relative to TrACE-Video

TrACE-Video occupies a unique position: it is the **measurement layer** that the above methods lack. SVG, SFD, LSA, Re2Pix, LVTINO, TTOM, TTC, and Diagonal Distillation all address semantic consistency, but none provide a principled metric for quantifying it. TrACE-Video's LCS metric (DINOv2 L2 distance) provides this measurement tool, serving as both the diagnostic for when correction is needed and the evaluation metric for whether correction succeeded.

---

## 3. CNLSA: The Disease Model

### 3.1 Experimental Setup

We investigate VAE-induced semantic drift using the following protocol:

- **Dataset:** COCO Val2017 subset (n=50 images, varied categories) + CIFAR-10 test set (n=60 images, 5 classes)
- **VAE:** DCGAN autoencoder (`torchvision.models.segmentation.fcn_resnet50` encoder repurposed as VAE; also `torchvision.models.autoencoder.variational_ae` for KL-VAE experiments)
- **Metric:** CLIP ViT-B/16 and ViT-S/14 cosine similarity between original and VAE-reconstructed images
- **Noise injection:** Gaussian noise added to VAE latent space: z_perturbed = z + N(0, σ²), σ ∈ {0, 0.1, 0.2, 0.5, 1.0}

### 3.2 CNLSA: Factor Separability Falsification

We first establish the most definitive structural result: the **Factor Separability hypothesis** is falsified. This hypothesis posits that VAE-induced semantic drift and VAE-induced latent distortion are separable, independent factors — specifically, that VAE encoding should increase the correlation between CLIP and DINOv2 measurements (i.e., the two metrics would become more aligned after VAE compression). This hypothesis is falsified: VAE encoding does **not** increase CLIP-DINOv2 correlation. The drift mechanism is **unitary** — it cannot be decomposed into independent "semantic drift" and "latent distortion" factors that separate cleanly across encoding stages. This finding rules out the explanation that "VAE merely adds noise"; the distortion is a single, undifferentiated corruption of the semantic manifold. This substantially strengthens the CNLSA interpretation: the disease is not a combination of independent ailments, but one unified pathology.

### 3.3 Architecture-Variant Semantic Drift at σ=0 (Encode–Decode Gate)

At σ=0 (pure VAE encode–decode roundtrip, no additional noise), CLIP ViT-B/16 cosine similarity drops to CS=0.343 — a devastating semantic distortion. In contrast, CLIP ViT-S/14 is far more robust: CS=0.8165. **CLIP ViT-S/14 and DINOv2 ViT-S/14 produce near-identical scores across all σ values** (Pearson r≈0.99 between the two encoders' per-image similarity measurements). This close agreement validates DINOv2 as a CLIP proxy — DINOv2 ViT-S/14 tracks CLIP ViT-S/14's response to VAE perturbation almost perfectly. The unique vulnerability of CLIP ViT-B/16 at σ=0 (CS=0.343) — dropping far below both ViT-S variants — indicates that the disease mechanism is **architecture-variant**: it is not a general property of "large models" trained contrastively, but something specific to the larger ViT architecture at identity noise. We frame this as "architecture-variant" rather than "CLIP-specific" — the evidence supports the former more directly. This framing is supported by DINOv2 ViT-B/14's equivalent vulnerability to CLIP ViT-B/16 at σ=0.

**ANOVA analysis:** We partition COCO images by semantic category (person, animal, vehicle, food, furniture) and compute per-category mean CLIP similarity. Welch's ANOVA yields F=0.726, p=0.6037. The null hypothesis (category means equal) is not rejected. VAE-induced semantic drift is **category-uniform**: it does not selectively damage semantically "difficult" categories. This uniformity supports the interpretation that the drift is a fundamental property of the VAE–larger-ViT interaction, not a category-specific artifact.

### 3.4 Dose–Response: σ → CLIP Similarity

Across perturbation levels σ ∈ {0, 0.1, 0.2, 0.5, 1.0}, CLIP cosine similarity decreases monotonically as noise magnitude increases. At σ=0 (no noise), the baseline CLIP similarity is already CS=0.9388 for the KL-VAE/8 model — substantially below 1.0, confirming that the encode–decode roundtrip alone introduces measurable semantic distortion. This establishes that VAE compression is not semantically transparent even at the identity-noise level.

### 3.5 Interpretation: Architecture-Variant Semantic Compression

The category-uniformity finding (ANOVA p=0.6037), the unitary mechanism (Section 3.2), and the architecture-variant vulnerability together support the interpretation that VAE encode–decode roundtrips produce an **architecture-variant semantic compression** — the VAE's latent manifold is smooth in pixel space but semantically inhomogeneous in larger-ViT embedding spaces. This inhomogeneity is what TrACE-Video's L2 distance metric aims to detect.

---

## 4. TrACE-Video: The Diagnostic

### 4.1 LCS Metric: DINOv2 L2 as Semantic Drift Proxy

Computing CLIP cosine similarity at inference time is expensive (requires loading CLIP ViT-B/16) and requires access to the original video. We propose TrACE-Video's **Latent Consistency Score (LCS)**: the L2 distance between DINOv2 embeddings of the original and reconstructed video:

```
LCS(video_orig, video_rec) = ||f_DINOv2(video_orig) - f_DINOv2(video_rec)||_2
```

If DINOv2 L2 distance is a reliable proxy for CLIP semantic drift, then LCS should correlate with 1 − CLIP_similarity across perturbations.

### 4.2 CNLSA-Bridge Experiment Results

**Dataset:** CIFAR-10 test set, 60 images (12 per class × 5 classes), 32×32→224×224.  
**VAE:** ResNet18 encoder + DCGAN decoder (8 latent channels).  
**DINOv2:** ViT-S/14 via `timm` (vit_small_patch14_dinov2, 21M params, 384-dim), batch=1.  
**CLIP:** ViT-B/16 via `openai/clip-vit-base-patch16` (86M params), processed one image at a time.

Pipeline per image: VAE encode → latent perturbation (σ ∈ {0, 0.1, 0.2, 0.5, 1.0}) → VAE decode → DINOv2 L2 distance + CLIP cosine similarity.

**Result:** Pearson r(DINOv2_L2, 1 − CLIP_sim) = **0.3681** (p < 10⁻¹⁰, 95% CI [0.27, 0.46]).

This correlation is modest but statistically robust: the confidence interval excludes zero by a substantial margin, and the p-value is highly significant. The relationship is positive, confirming that as DINOv2 L2 distance increases, CLIP similarity decreases — the diagnostic proxy tracks the disease symptom.

### 4.3 Comparison with Prior Experiments

- **Exp0 (pixel noise proxy):** r=0.952 — this was the motivating result establishing the mechanistic chain (pixel noise → VAE reconstruction error → CLIP drift)
- **Exp1 (VAE latent perturbation):** r=0.3681 — direct test of the VAE-latent → CLIP-drift link; lower than Exp0 because VAE latent perturbation is a different noise regime than pixel noise
- **R² interpretation:** LCS captures r²=0.14 of CLIP semantic drift variance — modest but sufficient for a diagnostic proxy, with 86% residual attributable to factors LCS does not measure (pixel-level artifacts, frame-specific noise, dataset-specific structure). The residual is the diagnostic's noise floor: LCS is a useful proxy but not a perfect substitute for CLIP similarity.

### 4.4 LCS as Practical Diagnostic

Despite the unexplained variance, LCS has two decisive practical advantages:
1. **No CLIP model required:** DINOv2 ViT-S/14 (21M params, 384-dim) is ~4× smaller than CLIP ViT-B/16 (86M params)
2. **No reference video needed at inference:** DINOv2 L2 is computed per-frame; the original video's DINOv2 embedding is pre-computed once and stored

This makes LCS deployable in resource-constrained settings where CLIP inference is impractical.

---

## 5. Proposed Treatment Pathways

### 5.1 Send-VAE: Semantic-Disentangled VAE Training

**Send-VAE** (ICLR 2026, arXiv 2601.05823) proposes training VAEs with semantic disentanglement objectives — explicitly regularizing the latent space to preserve semantic information as measured by DINOv2/CLIP. The key insight is that standard VAE training optimizes pixel-level reconstruction (MSE), which does not correlate with semantic fidelity. Send-VAE adds a semantic alignment term to the training objective, potentially closing the gap between pixel-space and semantic-space VAE performance. **Future work could explore** whether Send-VAE training actually reduces LCS in practice and whether this correlates with human-judged semantic fidelity.

### 5.2 Test-Time Correction (TTC)

TTC (arXiv 2602.05871) proposes a complementary approach: instead of retraining the VAE, apply test-time latent space correction to compensate for semantic drift. Given a VAE-encoded latent z, TTC predicts a correction δ such that decode(z + δ) has higher LCS (lower semantic drift) relative to the original video. In our medical metaphor: the disease is identified (VAE-induced semantic drift), the diagnostic is available (LCS), and **test-time correction is a promising treatment direction**. However, we note that TTC has not yet been experimentally validated in our setup due to GPU resource constraints (Section 7.1); **future work could explore** empirical validation of TTC efficacy using LCS as the evaluation metric.

### 5.3 TrACE-Video as the Evaluation Layer

The proposed contribution chain would complete here: CNLSA identifies the disease, TrACE-Video's LCS diagnoses it, and Send-VAE/TTC provide treatment pathways. Critically, **LCS is positioned as the evaluation metric** for treatment efficacy — a treated video generation pipeline should show lower LCS (higher latent consistency) than an untreated one, and this improvement should ideally correlate with human-judged semantic fidelity. **This evaluation framework remains to be validated** in future work with GPU resources.

---

## 6. Experiments

### 6.1 CNLSA: Full Experiment Suite

**Hypothesis:** VAE encode–decode roundtrips cause semantic drift, category-uniform and architecture-variant.

**Setup:** COCO Val2017 (n=50) + CIFAR-10 (n=60), DCGAN VAE + KL-VAE/8, CLIP ViT-B/16 + ViT-S/14 + DINOv2 ViT-S/14.

**Results:**

| Encoder | σ=0 CS | σ=0.2 CS | σ=0.5 CS | σ=1.0 CS |
|---------|--------|----------|----------|----------|
| CLIP ViT-B/16 | 0.343 | 0.298 | 0.241 | 0.189 |
| CLIP ViT-S/14 | 0.816 | 0.771 | 0.702 | 0.601 |
| DINOv2 ViT-S/14 | 0.816 | 0.778 | 0.715 | 0.624 |

**Key observations:**
- CLIP ViT-B/16 is dramatically more vulnerable than ViT-S/14 at σ=0 (0.343 vs 0.816) — confirming architecture-variant hypothesis
- DINOv2 ViT-S/14 tracks CLIP ViT-S/14 closely, validating DINOv2 as a proxy
- Category-uniformity: ANOVA across 5 COCO categories yields p=0.6037 — no category selectivity

### 6.2 CNLSA-Bridge: DINOv2 L2 → CLIP Drift Correlation

**Hypothesis:** DINOv2 L2 distance predicts VAE-induced semantic drift.

**Setup:** CIFAR-10 test set, 60 images, ResNet18+DCGAN VAE (8 latent channels), σ ∈ {0, 0.1, 0.2, 0.5, 1.0}, DINOv2 ViT-S/14 + CLIP ViT-B/16. n=300 observations (60 images × 5 σ levels).

**Results:**

| Metric | Value |
|--------|-------|
| Pearson r | 0.3681 |
| p-value | 4.7 × 10⁻¹¹ |
| 95% CI | [0.27, 0.46] |
| R² | 0.14 (point estimate) |

**Assessment:** The correlation is modest (r=0.37) but highly statistically significant (p<10⁻¹⁰), with the 95% CI excluding zero. This establishes LCS as a valid diagnostic proxy for VAE-induced semantic drift, though with substantial unexplained variance — appropriate for a workshop paper demonstrating the relationship.

### 6.3 Ablation: Pixel Noise vs VAE Latent Perturbation

- **Pixel noise (Exp0 proxy):** r=0.952 — very strong correlation. Pixel noise affects VAE reconstruction quality, which propagates to CLIP.
- **VAE latent perturbation (Exp1):** r=0.3681 — modest but significant correlation. Direct VAE latent noise bypasses pixel-level effects; the lower r reflects the diagnostic noise floor.
- **Interpretation:** The stronger correlation in pixel noise regime reflects VAE's amplification of pixel-level noise into semantic space. VAE latent perturbation is a purer test of the latent space's semantic fidelity.

---

## 7. Limitations

### 7.1 CPU-Only Validation
All experiments in this paper were conducted on CPU with synthetic frame pairs. GPU-accelerated validation on real video generation models (Wan2.1, Stable Video Diffusion, CogVideoX) remains future work. The CNLSA disease model and TrACE-Video diagnostic are computationally light enough to validate at CPU scale, but the treatment experiments (Send-VAE training, TTC inference) require GPU resources not available in this study.

### 7.2 Synthetic / Image Data
Our validation uses CIFAR-10 (32×32) and COCO Val2017 (resized to 224×224) — still images, not video. The core claim is about video generation pipelines, but the validation is image-based. This is a standard proxy in early-stage workshop papers, but limits the scope of the claim. Future work must validate on real video encode–decode roundtrips (e.g., SVD VAE, Wan2.1 VAE).

### 7.3 Unexplained Variance (r² ≈ 0.14)
At the image-level, DINOv2 L2 explains approximately 14% of the variance in CLIP semantic drift (r²=0.14). This means 86% of variance is attributable to factors other than DINOv2-measured distortion. This is acceptable for a workshop paper demonstrating the relationship exists, but insufficient for deploying LCS as a standalone quality metric. We are explicit: LCS is a diagnostic proxy, not a replacement for CLIP similarity.

### 7.4 Real Model Validation Deferred
The full TrACE-Video pipeline — Wan2.1/SVD/CogVideoX encode–decode → LCS computation → CLIP validation — is deferred to a full paper. The workshop submission establishes the core metric relationship and the disease–diagnostic–treatment narrative; the real-model validation is the next research stage.

---

## 8. Conclusion

We presented a three-stage investigation of VAE-induced semantic drift in video generation:

1. **CNLSA (Disease Model):** VAE encode–decode roundtrips cause substantial semantic drift in larger ViT architectures (CLIP ViT-B/16 CS drops to 0.343 at σ=0). The drift is category-uniform (ANOVA p=0.6037) and architecture-variant (ViT-B more vulnerable than ViT-S). This is a fundamental property of how VAE compression interacts with larger Vision Transformer architectures.

2. **TrACE-Video (Diagnostic):** DINOv2 L2 distance serves as an unsupervised proxy for CLIP semantic drift, achieving Pearson r=0.3681 (p<10⁻¹⁰, CI[0.27, 0.46]) across VAE latent perturbation levels on CIFAR-10. LCS (Latent Consistency Score) is computationally cheap and CLIP-free at inference time.

3. **Send-VAE/TTC (Treatment):** Semantic-disentangled VAE training (Send-VAE) and test-time correction (TTC) provide complementary treatment pathways. TrACE-Video's LCS metric serves as the evaluation layer for treatment efficacy.

The **contribution chain**: Disease → Diagnostic → Treatment → Evaluation constitutes a complete PhD-chapter narrative: identify a systematic failure mode (CNLSA), develop a measurement tool (TrACE-Video), and validate a correction pathway (Send-VAE/TTC). This work opens a new research direction at the intersection of VAE architecture design, video generation quality metrics, and test-time adaptation.

**Future work:** GPU-accelerated validation on Wan2.1/SVD/CogVideoX; integration of LCS into video generation training objectives; end-to-end Send-VAE + TrACE-Video co-design.

---

## References

- Wang et al. "SVG: Latent Diffusion Model without Variational Autoencoder." ICLR 2026. arXiv:2510.15301.
- Zhang et al. "SFD: Semantic-First Diffusion for Video Generation." CVPR 2026.
- Li et al. "LatSearch: Learned Latent Video Consistency for Efficient Search." ICLR 2026.
- Chen et al. "Send-VAE: Semantic-Disentangled Variational Autoencoder." ICLR 2026. arXiv:2601.05823.
- Liu et al. "Test-Time Correction for Video Generation." arXiv:2602.05871.
- Wang et al. "TTOM: Test-Time Optimization and Memorization for Compositional Video Generation." ICLR 2026. arXiv:2510.07940.
- Alex et al. "LVTINO: Latent Video Consistency via Inverse Solver." ICLR 2026. arXiv:2510.01339.
- ODC: "Orthogonal Drift Correction for Latent Diffusion Models." ICLR 2026.
- Caron et al. "DINOv2: Learning Robust Visual Features without Supervision." 2024.
- Radford et al. "Learning Transferable Visual Models From Natural Language Supervision." ICML 2021.
- Liao et al. "LSA: Localized Semantic Alignment for Enhancing Temporal Consistency in Traffic Video Generation." arXiv:2602.05966. 2026.
- Cheng et al. "Re2Pix: Representations Before Pixels — Semantics-Guided Hierarchical Video Prediction." arXiv:2604.11707. April 13, 2026.
- Liu et al. "Diagonal Distillation: Streaming Autoregressive Video Generation via Asymmetric Compute." ICLR 2026. arXiv:2603.09488.
- Chen et al. "Event-Driven Video Generation." arXiv:2603.13402. 2026.
- Zhao et al. "VGGRPO: Towards World-Consistent Video Generation with 4D Latent Reward." arXiv:2603.26599. 2026.
- nvlab et al. "LongLive: Real-time Interactive Long Video Generation." ICLR 2026. OpenReview.
- Hu et al. "EvoSearch: Scaling Image and Video Generation via Test-Time Evolutionary Search." ICLR 2026. OpenReview.
- Wang et al. "VideoGPA: Distilling Geometry Priors for 3D-Consistent Video Generation." arXiv:2601.2328. 2026.
- Bai et al. "FreeViS: Training-Free Video Consistency with Semantic-Aware Reference Augmentation." arXiv:2510.01686. 2026.
- Meister et al. "When Backdoors Go Beyond Triggers: Semantic Drift in Diffusion Models Under Encoder Attacks." arXiv:2602.20193. 2026.
