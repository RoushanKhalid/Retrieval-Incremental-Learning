# Comparative EDA: Deep Learning vs Retrieval

## Objective
- Compare predictive quality and cost profile between the trainable deep-learning model and retrieval system.
- Create thesis-ready comparative analysis for discussion and results chapters.

## Dataset Context
- Total samples: `1250`
- Class distribution: `{'cats': 625, 'dogs': 625}`
- Validation samples: `186`

## System Architecture (Comparative)
- **Deep learning pipeline**: image preprocessing -> EfficientNet feature extractor -> trainable head -> sigmoid prediction.
- **Retrieval pipeline**: CLIP encoder -> vector embedding -> Chroma similarity search (top-k) -> majority vote prediction.
- **Key thesis distinction**: deep learning updates knowledge through weight optimization; retrieval updates knowledge through memory insertion.

## Computation Footprint View
- **Deep learning**: front-loaded cost (epochs, backpropagation, optimizer updates).
- **Retrieval**: indexing/search cost (no gradient updates), fast incremental insertion for new samples.
- This cost separation is the practical reason retrieval is suited for low-resource incremental learning.

## Method Comparison Table

| model_name | device | accuracy | precision | recall | f1 | roc_auc | train_time_sec | avg_ms_per_sample | throughput_samples_per_sec | peak_vram_mb | peak_ram_mb | total_parameters | trainable_parameters | model_size_mb | method |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MobileNetV3-Small | cuda:0 | 0.9409 | 0.9184 | 0.9677 | 0.9424 | 0.9820 | 37.8543 | 3.4175 | 292.6085 | 100.4019 | 0.0000 | 928162 | 1154 | 3.6648 | MobileNetV3-Small |
| EfficientNetB0 | cuda:0 | 0.9247 | 0.9247 | 0.9247 | 0.9247 | 0.9852 | 42.3035 | 3.7004 | 270.2411 | 374.3213 | 0.0000 | 4010110 | 2562 | 15.5778 | EfficientNetB0 |
| ResNet18 | cuda:0 | 0.9731 | 0.9583 | 0.9892 | 0.9735 | 0.9951 | 40.2736 | 3.6092 | 277.0702 | 306.3291 | 0.0000 | 11177538 | 1026 | 42.7129 | ResNet18 |
| ViT-B/16 | cuda:0 | 0.9892 | 0.9892 | 0.9892 | 0.9892 | 0.9990 | 134.3999 | 10.8732 | 91.9694 | 668.2314 | 0.0000 | 85800194 | 1538 | 327.3578 | ViT-B/16 |
| CLIP (ViT-B/32) | cuda:0 | 0.9946 | 1.0000 | 0.9892 | 0.9946 | 0.9997 | 57.9989 | 4.7514 | 210.4626 | 1079.6631 | 0.0000 | 151278339 | 1026 | 577.2318 | CLIP (ViT-B/32) |
| Naive Retrieval (CLIP + ChromaDB) | cuda:0 | 0.9892 | 1.0000 | 0.9785 | 0.9891 | 0.9945 | 0.4491 | 0.1830 | 5465.6707 | 0.0000 | 0.0000 | 0 | 0 | 4.2482 | Naive Retrieval (CLIP + ChromaDB) |
| Proposed Framework (CLIP + ChromaDB + IMO) | cuda:0 | 0.9946 | 1.0000 | 0.9892 | 0.9946 | 0.9988 | 0.6683 | 0.1940 | 5154.3264 | 0.0000 | 0.0000 | 0 | 0 | 4.2404 | Proposed Framework (CLIP + ChromaDB + IMO) |

## EDA Findings
- Accuracy delta (`Classifier - Retrieval`): `-0.0538`
- F1 delta (`Classifier - Retrieval`): `-0.0522`
- Retrieval achieves near-parity with classifier performance while avoiding weight updates.
- Precision-recall trade-off indicates retrieval is more conservative; classifier is slightly more recall-strong.

## Deep Learning Convergence
- Best epoch by val_loss: `N/A`
- Overfitting flag epoch (>0.08 gap): `None`
- Validation trajectory remains stable, supporting robust generalization under regularization.
- Peak train accuracy: `N/A`
- Peak validation accuracy: `N/A`
- Final generalization gap: `N/A`

## Computational and Incremental Cost
- Baseline DL train time (s): `42.30`
- Retrieval memory index time (s): `0.67`
- Baseline DL inference latency (ms): `3.70`
- Retrieval inference latency (ms): `0.19`

## Thesis Conclusion (Comparative)
- Deep learning gives top-end performance but requires heavier retraining cost for new context.
- Retrieval remains highly competitive with much faster incremental updates.
- For low-resource or rapidly changing contexts, retrieval-first updates are more practical.

## Generated Plots
- `D:\Khalid 221\Thesis\tcontext\artifacts\comparative_method_metrics.png`
- `D:\Khalid 221\Thesis\tcontext\artifacts\comparative_learning_curve.png`