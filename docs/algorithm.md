# Algorithm: Classical CV Pedestrian & Vehicle Detection

## Overview

The system uses **one unified classical pipeline**. There is no neural network or trained model. Every step is explainable from first principles.

```
video frame
    │
    ▼
┌─────────────────────────────────────┐
│  1. Background Subtraction (MOG2)   │  → foreground mask (binary image)
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  2. Morphological Filtering          │  → cleaned mask
│     Opening  → removes noise speckles│
│     Closing  → fills holes in blobs  │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  3. Contour Extraction               │  → list of blob outlines
│     findContours (RETR_EXTERNAL)     │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  4. Area Filter                      │  → bounding boxes within range
│     min_area < blob < max_area       │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  5. Geometric Classifier             │  → labelled boxes (P / V)
│     aspect ratio + area rules        │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  6. Centroid Tracker                 │  → stable object IDs across frames
│     nearest-neighbour matching       │
└─────────────────────────────────────┘
    │
    ▼
OpenCV window + terminal stats
```

---

## Step-by-step Explanation

### 1. Background Subtraction — MOG2

**What it does:** Learns what the "empty" scene looks like, then flags pixels
that deviate from that learned background.

**How MOG2 works:**
Each pixel's intensity history over the last `history` frames is modelled as
a mixture of K Gaussian distributions (K = 3–5). At each new frame:
- Pixels whose value fits within one of the Gaussians → **background** (0)
- Pixels whose value doesn't fit → **foreground** (255)

The model updates continuously, so slow changes (cloud shadows, lights
turning on) are gradually absorbed into the background model.

**Parameters:**
- `bg_history` (default 300): larger = slower adaptation = better for static cameras
- `bg_var_threshold` (default 50): sensitivity; lower = more foreground pixels

**Why not frame differencing?**
Simple frame differencing (`|frame_t - frame_{t-1}|`) only catches pixels
that changed *between two consecutive frames*. A slowly-moving vehicle will
appear to have a "hole" in its centre. MOG2 maintains a history and avoids this.

---

### 2. Morphological Filtering

After background subtraction the mask contains:
- True foreground blobs (moving objects)
- Salt-and-pepper noise (sensor noise, small branches, reflections)
- Gaps inside large objects (e.g., a car's windshield matches the background)

Two morphological operations fix these problems:

**Opening** (erosion → dilation):
- Erosion shrinks bright regions; small noise specks disappear entirely.
- Dilation grows them back to original size.
- Net effect: removes objects smaller than the structuring element.

**Closing** (dilation → erosion):
- Dilation expands bright regions, merging nearby blobs and covering holes.
- Erosion shrinks them back.
- Net effect: fills holes and gaps inside objects.

Kernel shape: `MORPH_ELLIPSE` is preferred over a rectangle because real
objects have curved, not rectangular, boundaries.

---

### 3. Contour Extraction

`cv2.findContours` traces the boundary of each connected white region in the
binary mask. We use `RETR_EXTERNAL` (outermost contours only) because nested
contours would complicate counting without adding value.

`CHAIN_APPROX_SIMPLE` compresses collinear boundary points into endpoints only,
reducing memory usage — sufficient because we only need bounding rectangles.

---

### 4. Area Filter

Before classification, blobs are filtered by their bounding-box area:
- Too small (< `contour_min_area`) → residual noise after morphology
- Too large (> `contour_max_area`) → large background disturbance (trees, lighting)

This is the cheapest possible filter and removes most false positives.

---

### 5. Geometric Classifier

**Hypothesis:** In typical surveillance footage, pedestrians and vehicles have
systematically different bounding-box shapes.

| Class      | Aspect ratio (h/w) | Typical area (px²)   |
|------------|-------------------|----------------------|
| Pedestrian | ≥ 1.2 (tall)      | 600 – 12 000         |
| Vehicle    | < 1.2 (wide/sq.)  | 4 000 – 80 000       |

**Decision rules (in order):**
1. If `h/w ≥ 1.2` AND `area ≤ 12 000` → **pedestrian**
2. If `area ≥ 4 000` AND `h/w < 1.2`  → **vehicle**
3. Otherwise, use area as tie-breaker: smaller → pedestrian, larger → vehicle

**Known limitations:**
- Crouching pedestrian: low aspect ratio → may be classified as vehicle
- Distant vehicle: small area → may be classified as pedestrian
- Motorcycle: ambiguous geometry
- These cases are expected and documented; they should be mentioned in the demo.

---

### 6. Centroid Tracker

**Purpose:** Assign a stable ID to each object so it is counted once, not once
per frame.

**Algorithm:**
1. Represent each detected bounding box by its centroid (cx, cy).
2. Build an M×N pairwise Euclidean distance matrix between existing tracks
   and new detections.
3. Greedily match tracks to detections in order of closest distance.
4. Unmatched detections → new tracks with new IDs.
5. Unmatched tracks → increment a "disappeared" counter.
6. Tracks disappeared for > `max_disappeared` frames → delete.

**Why greedy instead of Hungarian algorithm?**
The Hungarian algorithm finds the globally optimal assignment but is O(n³)
and harder to explain in a demo. For typical traffic scenes with < 20
simultaneous objects, the greedy approach produces identical results.

---

## Performance on Jetson Orin Nano

| Component             | Typical time | Notes                                 |
|-----------------------|-------------|---------------------------------------|
| Frame read (CSI)      | ~5 ms       | GStreamer hardware decode              |
| MOG2 subtraction      | ~4 ms       | CPU; CUDA variant exists but not needed |
| Morphology            | ~1 ms       | Cheap for 720p                        |
| Contours + classify   | <1 ms       | Negligible                            |
| Centroid tracker      | <1 ms       | NumPy distance matrix                 |
| Display (imshow)      | ~5 ms       | X11 or HDMI                           |
| **Total (720p)**      | **~15 ms**  | **~60 FPS headroom**                  |

Actual FPS reported at runtime depends on input resolution and frame rate.
