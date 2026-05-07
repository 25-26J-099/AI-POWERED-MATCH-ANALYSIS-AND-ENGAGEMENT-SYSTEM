# AI-Powered Match Analysis for Low-Resource Football: A Lightweight Video Pipeline for Tracking, Re-Identification, Event Detection, and Structured Data Export

Muaad M F M  
Department of Computer Science  
Sri Lanka Institute of Information Technology  
IT22323620

## Abstract

Grassroots, school-level, and lower-league football matches are rarely supported by the camera infrastructure, tracking systems, and commercial analytics platforms available in professional football. This creates an analytics gap in which coaches, players, and local communities lack objective match intelligence despite having access to basic single-camera video footage. This paper presents a lightweight match-analysis pipeline designed for low-resource football environments. The proposed component transforms low-quality single-camera match video into structured football data through a sequence of computer-vision modules: video preprocessing, player and ball detection, multi-object tracking, team color assignment, player re-identification, hybrid event detection, and StatsBomb-style export. The system combines YOLOv8n-based object detection, ByteTrack-inspired tracking, HSV/KMeans team assignment, OCR-supported jersey recognition, robust re-identification backends, rule-based event logic, optional machine-learning event classification, and structured JSON/CSV export. Unlike downstream commentary or decision-intelligence components, this contribution functions as the upstream perception and event-generation backbone of the overall football analysis platform. The implemented prototype demonstrates an end-to-end path from raw video upload to machine-readable events, player trajectories, possession logs, freeze-frame records, and analytics-ready StatsBomb-style outputs. The paper also discusses the component's limitations, including single-camera visibility constraints, occlusion sensitivity, lighting variation, and the need for broader quantitative validation on real amateur-football datasets.

Index Terms - football analytics, computer vision, low-resource football, player tracking, ball tracking, re-identification, event detection, team assignment, StatsBomb export

## I. Introduction

Modern football analytics has become increasingly data-driven. Professional clubs and broadcasters use multi-camera tracking systems, event-data providers, wearable sensors, and proprietary platforms to evaluate tactical structure, player performance, and match momentum. However, these technologies remain largely inaccessible to grassroots clubs, schools, amateur leagues, and local tournaments. In many low-resource football environments, the only available data source is a single low-cost camera recording from the touchline, often affected by camera shake, low resolution, motion blur, variable lighting, and partial field visibility.

The central research problem addressed by this component is how to convert such low-quality single-camera football footage into structured match intelligence without requiring professional broadcast infrastructure. A usable system must detect and track players and the ball, maintain player identities across frame exits or occlusions, assign players to teams, identify meaningful events, and export the results in a format that downstream analytics and commentary modules can consume.

This paper proposes and documents a lightweight AI-powered match-analysis pipeline for low-resource football games. The component is designed as the perception and event-generation layer of a larger AI-powered match analysis and engagement system. It does not primarily generate commentary, audience adaptation, or decision-intelligence ratings. Instead, it produces the machine-readable tracking and event foundation on which those later components depend.

The main contributions of this component are:

1. A preprocessing layer for stabilizing and enhancing low-quality single-camera video.
2. A lightweight player and ball tracking pipeline using YOLOv8n-style detection and ByteTrack-inspired association.
3. A team assignment module based on jersey color clustering, with frontend-supported mapping from detected team colors to real team names.
4. A re-identification layer that preserves player identity through occlusion, exit, and re-entry scenarios.
5. A hybrid event detection engine combining deterministic rules with optional machine-learning classification.
6. A structured export pipeline that generates JSON, CSV, freeze-frame, and StatsBomb-style artifacts for analytics and commentary.

The remainder of this paper is organized as follows. Section II reviews related work. Section III describes the proposed methodology and system architecture. Section IV presents the implementation and validation approach. Section V discusses limitations and future work. Section VI concludes the paper.

## II. Related Work

### A. Object Detection in Sports Video

Object detection is the entry point for most automated sports analysis systems. Modern football perception pipelines commonly rely on one-stage detectors such as YOLO because of their speed and practical deployment characteristics. YOLOv8n is especially suitable for low-resource environments because it offers a favorable trade-off between inference speed and detection quality. In this component, YOLOv8n-style detection is used to identify players and the ball in each video frame, while fallback paths are retained for environments where heavier deep-learning dependencies are unavailable.

### B. Multi-Object Tracking

Tracking-by-detection systems associate detections across frames to create continuous player trajectories. ByteTrack is particularly relevant for low-quality footage because it retains both high-confidence and low-confidence detections during association. This is useful in amateur football video, where motion blur, occlusion, and lighting changes often reduce detector confidence even when the detection is visually correct. The proposed component adopts a ByteTrack-inspired association strategy and extends it with motion and appearance cues to reduce identity switches during player crossings.

### C. Player Re-Identification

Standard tracking methods struggle when a player leaves the camera view and later re-enters. Player re-identification attempts to match new detections to previously observed identities using appearance features, spatial history, and temporal context. Sports Re-ID is more difficult than pedestrian Re-ID because players on the same team wear visually similar kits, are frequently occluded, and appear at low resolution. This component uses a backend ladder for Re-ID: FastReID where available, torchreid as a practical fallback, and handcrafted embeddings as a safe fallback when deep Re-ID backends are not available.

### D. Team Assignment by Kit Color

Team assignment is necessary for possession, event attribution, tactical analysis, and readable downstream outputs. In low-resource footage, team identity can often be inferred from jersey color. The proposed module extracts torso-region color features from tracked player crops, filters pitch-green pixels, clusters the remaining HSV color samples, and assigns players to detected team clusters. Since color clusters are anonymous, the frontend includes a confirmation step where users map detected colors such as "Orange" or "Blue" to real team names.

### E. Event Detection and Structured Football Data

Football event detection transforms visual observations into semantic actions such as passes, shots, carries, tackles, pressures, recoveries, and possession changes. Purely rule-based methods are interpretable and efficient but can miss complex temporal patterns. Pure machine-learning methods can capture richer patterns but require labeled data and may be harder to validate. The proposed component therefore uses a hybrid approach: deterministic rules handle geometry-driven events, while optional machine-learning detection supports temporally complex events. The final outputs are exported as structured JSON, CSV, and StatsBomb-style events so that analytics and commentary components can consume them consistently.

## III. Methodology

The proposed system is a sequential but modular pipeline that transforms raw match video into structured football data. Figure 1 conceptually summarizes the processing flow:

Raw video -> preprocessing -> detection -> tracking -> team assignment -> Re-ID -> event detection -> structured export -> analytics/commentary handoff

### A. Video Preprocessing

Low-resource match footage often contains camera shake, poor resolution, uneven exposure, and compression artifacts. The preprocessing module improves input quality before detection and tracking. The implementation includes:

- frame resizing for resource control;
- optional stabilization to reduce camera motion;
- contrast and visual enhancement;
- super-resolution fallback paths where configured;
- frame-skip controls for speed-sensitive deployment.

The preprocessing module is not intended to create broadcast-quality video. Its purpose is to improve downstream model reliability while keeping computational requirements manageable.

### B. Player and Ball Detection

The detection module identifies player and ball candidates in each frame. The primary backend uses YOLOv8n-style inference because the nano model is suitable for consumer hardware and low-latency processing. Class-specific confidence thresholds are applied: player detections use a stricter threshold, while ball detections use a lower threshold because the ball is smaller, frequently blurred, and more difficult to detect reliably.

When the preferred detector is unavailable, the system can fall back to OpenCV-based alternatives. This fallback design supports the low-resource objective by allowing the pipeline to continue operating even when the full deep-learning stack is not installed.

### C. Multi-Object Tracking

The tracking module converts frame-level detections into temporally consistent player tracks. It follows a ByteTrack-inspired two-stage association strategy:

1. High-confidence detections are matched to active tracks.
2. Lower-confidence detections are considered for unmatched tracks.

This design is useful in low-quality footage because valid player detections may receive low confidence during blur or occlusion. The custom association logic uses a hybrid cost function that combines:

- spatial overlap through intersection-over-union;
- motion prediction from recent trajectory history;
- appearance similarity from jersey-region color histograms;
- team consistency constraints once team assignment is available.

The ball is tracked separately using a robust tracker with temporal smoothing, because ball movement and visibility patterns differ significantly from player movement.

### D. Team Color Assignment

The team assignment module extracts jersey color from the torso region of tracked player bounding boxes. To prevent the green pitch from dominating the color estimate, HSV pixels within pitch-green ranges are filtered before clustering. KMeans clustering is then applied to the remaining color samples, producing two dominant team-color clusters.

The system originally labeled these clusters generically as Team 1 and Team 2. However, detected clusters are anonymous: users cannot know before upload which real-world team will map to which cluster. To solve this ambiguity, the updated workflow performs a quick team-color preview after upload. The frontend displays detected color groups, for example:

- Detected Team 1 - Orange
- Detected Team 2 - Blue

The user then assigns real team names to those detected color groups. The saved mapping is used throughout the full pipeline:

```json
{
  "0": "Colombo Lions",
  "1": "Kandy Blues"
}
```

This ensures that outputs, analytics, commentary, and frontend labels refer to real team names instead of generic cluster identifiers.

### E. Jersey OCR

Jersey number recognition is included as a supporting identity cue. The OCR module processes tracked player crops and applies preprocessing variants such as scaling, thresholding, denoising, and contrast enhancement. OCR results are aggregated over time so that unstable single-frame predictions do not immediately overwrite more stable jersey-number estimates. This is especially important in amateur footage, where jerseys are small, blurred, and frequently occluded.

### F. Player Re-Identification

The Re-ID subsystem maintains identity continuity when players leave the frame, become occluded, or re-enter after a tracking gap. The implementation includes:

- a robust identity manager;
- gallery-based appearance matching;
- temporal identity voting;
- team-aware constraints to reduce cross-team reassignment;
- backend resolution across FastReID, torchreid, and handcrafted embeddings.

The system is designed to degrade gracefully. If FastReID is unavailable, the pipeline can use torchreid. If deep backends are unavailable, handcrafted embeddings keep the pipeline operational, although with lower expected identity robustness.

### G. Hybrid Event Detection

The event detection layer converts tracked positions, ball state, possession state, and motion patterns into football events. Events are divided into routing categories:

- Rule-only events: geometry-driven events such as out-of-bounds, possession change, pressure, block, clearance, and goalkeeper actions.
- Machine-learning-supported events: temporal events such as pass, dribble, sprint, and interception where learned sequence patterns can help.
- Hybrid events: events such as tackle, foul, and carry where both rule context and learned classification are useful.

This hybrid design balances interpretability, computational efficiency, and extensibility. It also supports operation when machine-learning event weights are unavailable by preserving rule-based fallback behavior.

### H. Freeze Frames and Structured Export

The pipeline exports multiple artifact types:

- annotated tracking video;
- complete match-analysis JSON;
- event-only JSON;
- player trajectory CSV;
- freeze-frame JSON;
- StatsBomb-style event JSON.

The StatsBomb-style export normalizes event structure for downstream analytics. It includes event type, timestamp, period, team, player, location, action-specific metadata, possession information, and freeze-frame context where available. This makes the component compatible with later modules such as xT/xG/VAEP analytics, tactical commentary, and player-comparison dashboards.

## IV. Implementation and Validation

### A. Repository Implementation

The implemented component is located in the backend analysis layer of the project. The main implementation files are summarized in Table I.

Table I. Implementation Evidence

| Module | File path | Purpose |
| --- | --- | --- |
| Pipeline orchestration | `app/event_detection/pipeline.py` | Runs preprocessing, tracking, Re-ID, events, export |
| Video preprocessing | `app/event_detection/video_preprocessor.py` | Stabilization and enhancement |
| Player/ball tracking | `app/event_detection/tracker.py` | Detection and tracking |
| Team assignment | `app/event_detection/team_assigner.py` | Jersey color extraction and clustering |
| Team color preview | `app/services/team_color_service.py` | Pre-processing color detection for user mapping |
| Jersey OCR | `app/event_detection/jersey_ocr.py` | Jersey-number extraction |
| Re-ID | `app/event_detection/robust_reid.py`, `app/event_detection/reid_module.py` | Identity continuity |
| Event detection | `app/event_detection/strategic_hybrid_detector.py` | Rule/ML/hybrid event routing |
| StatsBomb export | `app/event_detection/statsbomb_export.py` | Analytics-ready structured event export |
| API exposure | `app/routes/video.py`, `app/routes/upload.py` | Upload and analysis endpoints |
| Analytics handoff | `app/services/merged_pipeline_service.py` | Connects Component 1 to downstream analytics |

### B. End-to-End Workflow

The implemented workflow is:

1. User uploads a match video.
2. The backend creates a match record and stores the raw video.
3. A quick team-color detection pass samples the video and returns anonymous detected colors.
4. The user maps detected color groups to real team names.
5. Optional lineups and formations are submitted.
6. The full Component 1 pipeline runs tracking and event detection.
7. Structured artifacts are exported.
8. Downstream analytics and commentary components consume the exported data.

This workflow resolves the team-name ambiguity by ensuring that users label detected color groups after the system has shown what those groups are.

### C. Functional Validation

The repository includes focused tests for backend services, event parsing, API job flow, Re-ID behavior, OCR behavior, StatsBomb export, tactical integration, and team-color metadata. The current implementation has been verified through:

- API job lifecycle tests;
- event parser tests;
- team color metadata tests;
- frontend production build;
- successful backend import and route registration;
- manual server startup and health-check validation.

These tests validate integration behavior and output contracts. However, they do not yet constitute a full scientific benchmark for detection accuracy, tracking accuracy, Re-ID identity switches, or event-detection precision/recall on a large real-world amateur-football dataset.

### D. Evaluation Metrics for Future Benchmarking

For a complete research evaluation, the following metrics should be collected on annotated low-resource football clips:

Table II. Recommended Evaluation Metrics

| Subsystem | Recommended metrics |
| --- | --- |
| Player detection | Precision, recall, mAP |
| Ball detection | Precision, recall, small-object recall |
| Multi-object tracking | MOTA, MOTP, IDF1, identity switches |
| Re-ID | Rank-1 accuracy, mAP, re-entry recovery rate |
| Team assignment | Team classification accuracy, color-cluster purity |
| Event detection | Precision, recall, F1 by event type |
| Runtime | FPS, average latency, hardware utilization |
| Export quality | Schema validity, downstream ingestion success |

The implemented system is ready to support such evaluation because it already produces frame-level trajectories, events, freeze frames, and structured outputs.

## V. Discussion

### A. Strengths

The main strength of the proposed component is its complete end-to-end integration. It does not stop at object detection or tracking; it connects raw video input to structured football data that can be used by analytics, commentary, and visualization modules. The modular design also allows individual subsystems to be improved independently.

The system is practical for low-resource football because it includes lightweight models, fallback paths, frame-skip controls, and CPU-compatible execution. The new team-color confirmation workflow also improves usability by preventing users from incorrectly assigning team names before knowing which color cluster corresponds to which team.

### B. Limitations

The component has several limitations:

1. Single-camera field of view prevents observation of off-frame players and events.
2. Occlusions remain difficult during dense player clustering.
3. Similar kits can reduce team assignment and Re-ID reliability.
4. Ball detection is sensitive to blur, scale, and camera distance.
5. Event detection is constrained by visible evidence and may miss tactical intent.
6. Formal real-match evaluation is still limited.
7. A released annotated amateur-football dataset is not yet packaged with the repository.

These limitations are expected in low-resource single-camera analysis and should be clearly stated in any academic or viva presentation.

### C. Future Work

Future work should focus on:

- collecting and annotating a representative amateur-football video dataset;
- benchmarking detection, tracking, Re-ID, and event-detection performance;
- improving ball tracking through temporal super-resolution or specialized ball detectors;
- adding active-learning workflows for correcting events and identities;
- improving team assignment for visually similar kits;
- integrating homography estimation for pitch-coordinate normalization;
- measuring runtime performance on consumer-grade laptops and low-end GPUs.

## VI. Conclusion

This paper presented a lightweight AI-powered match-analysis pipeline for low-resource football video. The component addresses the upstream perception problem in grassroots football analytics: converting single-camera footage into structured match data. The system integrates video preprocessing, player and ball tracking, team color assignment, Re-ID, hybrid event detection, freeze-frame extraction, and StatsBomb-style export. It also introduces a user-confirmed team-color mapping workflow to resolve ambiguity between anonymous detected color clusters and real-world team names. The implemented prototype provides a practical foundation for downstream analytics, player evaluation, commentary generation, and fan engagement. While broad quantitative validation remains future work, the repository demonstrates a functioning end-to-end pipeline and a research-safe basis for further evaluation.

## References

[1] K. Singh, "Introducing Expected Threat (xT)," football analytics research note, 2019.

[2] T. Decroos, L. Bransen, J. Van Haaren, and J. Davis, "Actions Speak Louder than Goals: Valuing Player Actions in Soccer," KDD, 2019.

[3] A. Bewley, Z. Ge, L. Ott, F. Ramos, and B. Upcroft, "Simple Online and Realtime Tracking," ICIP, 2016.

[4] Y. Zhang et al., "ByteTrack: Multi-object tracking by associating every detection box," ECCV, 2022.

[5] G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLO," 2023.

[6] N. Wojke, A. Bewley, and D. Paulus, "Simple Online and Realtime Tracking with a Deep Association Metric," ICIP, 2017.

[7] L. Zheng, Y. Yang, and A. G. Hauptmann, "Person Re-Identification: Past, Present and Future," arXiv preprint, 2016.

[8] StatsBomb, "StatsBomb Open Data and 360 Freeze Frames," public football event-data resource.

[9] OpenCV, "OpenCV Documentation: Video, Image Processing, and DNN Modules."

[10] Muaad M F M, "AI-Powered Match Analysis and Engagement System for Low-Resource Football Games: Component 1 Proposal," SLIIT project proposal, 2025.
