# Multimodal-Unit
Official implementation of Multi-Modal UNIT (MMU) — a unified transformer for joint language–vision understanding and content generation.
# 🧠 Multi-Modal UNIT (MMU)

Official implementation of the paper:  
**"Unified Transformer Framework for Integrated Language–Vision Understanding and Content Generation"**  
Submitted to *The Visual Computer* (Springer, 2025)

---

## 🔍 Overview
**Multi-Modal UNIT (MMU)** is a unified transformer architecture designed to bridge visual and linguistic understanding within a **single-stream framework**.  
Unlike traditional dual-encoder approaches that treat text and image separately, MMU introduces **lightweight modality adapters** that project visual and textual embeddings into a shared representational space.  
Through **shared attention layers** and a **hybrid optimization strategy** combining contrastive and generative objectives, MMU achieves both strong accuracy and high computational efficiency across multimodal tasks.

---

## 🧩 Key Contributions
- ✅ **Unified single-stream transformer** for both language and vision processing.  
- 🧠 **Lightweight modality adapters** for efficient feature alignment.  
- 🔄 **Hybrid objectives** integrating contrastive (understanding) and generative (captioning/reasoning) training.  
- 📊 **Consistent performance** across COCO, Flickr30k, VQAv2, and NLVR2 benchmarks.  
- ⚡ **210M parameters** with **70 ms per-sample inference latency**, achieving a strong accuracy–efficiency balance.

---

