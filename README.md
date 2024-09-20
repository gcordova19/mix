# Mixture of Experts (MoE) for Image Classification

This repository implements various Mixture of Experts (MoE) architectures for image classification, focusing on the CIFAR-10 dataset. The models are built on simple neural networks, Vision Transformers (ViT), and Reinforcement Learning (RL) to optimize expert selection.

### Key Components:
1. **MoE.py**: Basic MoE model with expert selection using a gating layer and simple output aggregation. Designed to integrate with Vision Transformers.
2. **MoeRandomSearch.py & Moecontensorboard.py**: Enhancements include TensorBoard integration for tracking metrics, visualization of training, and random hyperparameter search.
3. **MoeRL.py**: Adaptive MoE using RL for expert selection. The model aims to optimize expert layers but currently shows limited accuracy due to its simple architecture.
4. **MoeVit.py**: MoE with a pretrained ViT, incorporating a sparse-gated architecture and noisy gating mechanism for balanced expert selection.

### Improvements:
- Advanced data augmentation techniques.
- Integration of CNNs/Transformers as experts.
- Hyperparameter optimization and training adjustments.

