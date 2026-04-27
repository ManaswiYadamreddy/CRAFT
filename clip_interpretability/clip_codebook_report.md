# Text-to-Codebook Semantic Alignment

Regions: `eyes, skin, hair, lips, bg`  |  n_images=500  |  top_k=16

## Diagonal Dominance

- Mean-max (text→codebook) : 80% rows win  avg margin=+0.0161
- Centroid↔centroid        : 80% rows win  avg margin=+0.0153

## Text → Codebook Mean-Max Cosine Similarity

| src \ tgt | eyes | skin | hair | lips | bg |
|---|---|---|---|---|---|
| **eyes** | **0.2775** | 0.2644 | 0.2559 | 0.2385 | 0.2374 |
| **skin** | 0.2627 | **0.2871** | 0.2630 | 0.2556 | 0.2458 |
| **hair** | 0.2615 | 0.2585 | **0.2845** | 0.2285 | 0.2400 |
| **lips** | 0.2524 | 0.2738 | 0.2516 | **0.2965** | 0.2402 |
| **bg** | 0.2274 | 0.2353 | **0.2438** | 0.2161 | 0.2412 |

## Text Centroid ↔ Codebook Centroid Cosine

| src \ tgt | eyes | skin | hair | lips | bg |
|---|---|---|---|---|---|
| **eyes** | **0.2991** | 0.2791 | 0.2730 | 0.2548 | 0.2477 |
| **skin** | 0.2860 | **0.3029** | 0.2795 | 0.2761 | 0.2499 |
| **hair** | 0.2790 | 0.2797 | **0.3052** | 0.2508 | 0.2519 |
| **lips** | 0.2621 | 0.2869 | 0.2614 | **0.3078** | 0.2440 |
| **bg** | 0.2507 | 0.2554 | **0.2610** | 0.2399 | 0.2540 |
