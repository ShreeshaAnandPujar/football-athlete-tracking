# Detection Quality vs. Tracking (Association) Quality

A critical analytical finding in this pipeline is the decoupling of **detection performance** from **temporal association performance**.

In modern multi-object tracking (MOT), overall tracking performance as measured by **HOTA** (*Higher Order Tracking Accuracy*) is conventionally related to the geometric mean of detection accuracy ($\text{DetA}$) and association accuracy ($\text{AssA}$):

$$ \text{HOTA} \approx \sqrt{\text{DetA} \times \text{AssA}} $$

---

## 1. Architectural Pipeline Decomposition

```
Raw Video Stream / Frames
       │
       ▼
┌───────────────────────────────────────────────────────────┐
│ 1. DETECTION STAGE (Spatial Localization)                 │
│    • Model: Fine-tuned YOLOv8s on SoccerNet Tracking-2023 │
│    • Input Resolution: 640×640                             │
│    • Confidence Threshold: conf = 0.35                    │
│    • Output: 2D Spatial Bounding Boxes [x1, y1, x2, y2, c] │
└───────────────────────────────────────────────────────────┘
       │
       ▼
┌───────────────────────────────────────────────────────────┐
│ 2. TRACKING / ASSOCIATION STAGE (Temporal Continuity)      │
│    • Tracker: BoT-SORT                                    │
│    • Camera Motion Compensation: ORB Keypoint Homography  │
│    • State Prediction: Kalman Filter (8D state)           │
│    • Association Gating: match_thresh = 0.85 (IoU Cost)   │
│    • Memory Retention: track_buffer = 30 frames           │
│    • Output: Unique Persistent Trajectories (Track IDs)   │
└───────────────────────────────────────────────────────────┘
       │
       ▼
┌───────────────────────────────────────────────────────────┐
│ 3. EVALUATION HARNESS (SoccerNet TrackEval)                │
│    • Benchmark: 12 Validation Sequences (9,000 frames)    │
│    • Metrics: HOTA, DetA, AssA, IDF1, MOTA, IDSW, Frag     │
└───────────────────────────────────────────────────────────┘
```

---

## 2. Empirical Benchmark Comparison

From our official frozen benchmark evaluation:

| Metric Category | Metric | Score | Interpretation |
| :--- | :--- | :---: | :--- |
| **Detection Quality** | **DetA** | **70.489** | High spatial localization accuracy |
| | **CLEAR Precision ($\text{CLR\_Pr}$)** | **96.812%** | False positive detections are extremely rare (< 3.2%) |
| | **CLEAR Recall ($\text{CLR\_Re}$)** | **90.225%** | Detects 9 out of 10 players on pitch consistently |
| | **MOTA** | **86.829** | Very high baseline detection-weighted MOT score |
| **Tracking Quality** | **AssA** | **52.204** | Significant gap compared to DetA ($\Delta \approx -18.28$) |
| | **IDF1** | **70.615** | Global trajectory identity preservation |
| | **Association Precision ($\text{AssPr}$)** | **78.448%** | When matched, track associations are largely correct |
| | **Association Recall ($\text{AssRe}$)** | **55.825%** | Primary bottleneck: tracks fragment during prolonged occlusions |
| | **Identity Switches ($\text{IDSW}$)** | **693** | Total identity swaps across 9,000 frames |
| **Composite Metric** | **HOTA** | **60.618** | Conventionally related to the geometric mean of DetA and AssA ($\text{HOTA} \approx \sqrt{\text{DetA} \times \text{AssA}}$) |

---

## 3. Why Association is the Primary Bottleneck in Football

1. **Teammate Visual Homogeneity**:
   - Outfield teammates wear identical jerseys, shorts, and socks.
   - Standard ReID feature embeddings produce near-zero cosine distance between distinct teammates, causing identity swaps when they cross paths.

2. **Occlusion Scrums (Corners, Set-Pieces, Penalty Area Clusters)**:
   - In crowded penalty box events, multiple players overlap ($> 45\%\text{ of frames}$ exhibit overlaps with $\text{IoU} > 0.20$).
   - Pure spatial matching degrades when players enter occluded clusters and re-emerge in swapped positional order.

3. **Dynamic Camera Panning vs. Rapid Sprints**:
   - When the broadcast camera rapidly pans to follow a counter-attack, player screen velocities combine player sprint vectors with camera pan vectors.
   - Without GMC (Global Motion Compensation), Kalman filters fail. ORB GMC lifts AssA from **47.11** (without GMC) to **51.39** (+4.28 AssA points).

4. **Summary**:
   - Detection is currently less limiting than association on the evaluated SoccerNet benchmark. YOLOv8s achieves $\text{DetA} = 70.49$ while $\text{AssA} = 52.20$, indicating that association errors account for a larger share of the current HOTA limitation.
   - Potential future directions include jersey-number recognition and topology/formation-aware association.
