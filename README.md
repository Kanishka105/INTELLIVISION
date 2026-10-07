                         🚦 INTELLIVISION
          Integrated Computer Vision & Traffic Analytics
                                │
                                ▼
                    ┌─────────────────────┐
                    │   DATA / INPUT      │
                    │ MRTMD + Traffic     │
                    │ Images + Videos     │
                    └──────────┬──────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ LAYER 1 — IMAGE ACQUISITION & REPRESENTATION               │
│                                                             │
│ Raw Image                                                   │
│   ↓                                                         │
│ Pixels → Channels → Color Spaces → Resolution → ROI        │
│                                                             │
│ Why?                                                        │
│ Understand exactly what an image contains before processing │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ LAYER 2 — IMAGE PROCESSING & ENHANCEMENT                   │
│                                                             │
│ Resize → Noise Removal → Blur → Histogram → Equalization   │
│ → Thresholding → Edge Detection                            │
│                                                             │
│ Why?                                                        │
│ Make important information easier for algorithms to detect  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ LAYER 3 — SEGMENTATION & OBJECT/REGION ANALYSIS            │
│                                                             │
│ Thresholding → Segmentation → Morphology → Contours        │
│ → Connected Components → Regions                           │
│                                                             │
│ Why?                                                        │
│ Separate useful regions/objects from the background        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ LAYER 4 — FEATURES & FREQUENCY DOMAIN                       │
│                                                             │
│ Spatial Features                                             │
│   ├─ Shape                                                  │
│   ├─ Texture                                                │
│   └─ Color                                                  │
│                                                             │
│ Feature Descriptors                                         │
│   ├─ ORB                                                   │
│   ├─ SIFT                                                  │
│   └─ Hu Moments                                             │
│                                                             │
│ Frequency                                                   │
│   ├─ DFT                                                   │
│   └─ DCT                                                   │
│                                                             │
│ Why?                                                        │
│ Convert visual information into useful mathematical         │
│ features                                                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ LAYER 5 — AI VISION                                         │
│                                                             │
│              ┌───────────────┴───────────────┐              │
│              ▼                               ▼              │
│        YOLO / Detection                ANPR / OCR            │
│              │                               │              │
│        Vehicle Detection              Plate Detection       │
│        Person Detection                Plate Crop           │
│        Bus/Truck/Bike                   ↓                  │
│              │                          OCR                  │
│              │                           ↓                   │
│              │                     License Number            │
│              └───────────────┬───────────────┘              │
│                              ▼                              │
│                    Detected Information                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ LAYER 6 — VIDEO INTELLIGENCE & TRACKING                     │
│                                                             │
│ Video Frames → YOLO → ByteTrack → Object IDs               │
│                              ↓                              │
│                         Centroids                           │
│                              ↓                              │
│                        Trajectories                         │
│                              ↓                              │
│                 Direction / Movement                        │
│                              ↓                              │
│              Counting / Density / Flow                      │
│                              ↓                              │
│                  Approx. Speed                              │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ LAYER 7 — TRAFFIC ANALYTICS                                │
│                                                             │
│ Vehicle Count                                               │
│ Traffic Density                                              │
│ Vehicle Classification                                      │
│ Traffic Flow                                                │
│ Speed Estimation                                            │
│ Direction Analysis                                          │
│ Congestion Analysis                                         │
│ OCR / Number Plate Results                                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ LAYER 8 — DASHBOARD / VISUALIZATION                         │
│                                                             │
│ Streamlit / Web Dashboard                                   │
│                                                             │
│ ├─ Live/processed video                                    │
│ ├─ Bounding boxes                                           │
│ ├─ Vehicle count                                            │
│ ├─ Traffic density                                          │
│ ├─ Speed                                                    │
│ ├─ Vehicle classes                                          │
│ ├─ Number plates                                            │
│ └─ Graphs / statistics                                      │
└─────────────────────────────────────────────────────────────┘