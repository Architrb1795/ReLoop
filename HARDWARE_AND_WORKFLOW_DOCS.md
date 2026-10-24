# EcoVend AI — Smart Campus Waste Segregation RVM & Student Ecosystem
**Complete Hardware, AI Model, Dataset Evaluation & End-to-End Execution Documentation**

---

## 1. System Overview & Core Architecture

**EcoVend AI** is a closed-loop **Smart Reverse Vending Machine (RVM)** designed for hostel canteens and college campuses. Students dispose of waste items (either one-by-one or as a collective mixed batch into a singulating hopper), the machine optically classifies and physically segregates each item using an **AI Vision + ESP32 Servo Actuator pipeline**, and the student scans a **Dynamic Encrypted Session QR Code** on the machine screen using the **EcoVend Student Mobile App** to claim reward points redeemable at campus canteens.

### End-to-End Physical & Digital Workflow

```text
   [STUDENT DISPOSES MIXED OR SINGLE WASTE AT CANTEEN RVM]
                              │
                              ▼
       ┌─────────────────────────────────────────────┐
       │ 1. HOPPER & VIBRATORY SINGULATOR FEEDER     │
       │    Separates collective dump into 1-by-1    │
       └──────────────────────┬──────────────────────┘
                              │
                              ▼
       ┌─────────────────────────────────────────────┐
       │ 2. CONVEYOR BELT & ANTI-CHEAT CHAMBER       │
       │    • 1080p Camera + 5600K LED Dome          │
       │    • VL53L0X 3D ToF Sensor (Liveness Check) │
       │    • HX711 5kg Load Cell (Mass Plausibility)│
       └──────────────────────┬──────────────────────┘
                              │
                              ▼
       ┌─────────────────────────────────────────────┐
       │ 3. AI VISION CLASSIFIER (ResNet50 / YOLOv8) │
       │    • 224×224 RGB Normalized Tensor          │
       │    • Official Test Accuracy: 91.05%         │
       │    • Single-Image Inference Latency: 7.27ms │
       └──────────────────────┬──────────────────────┘
                              │
                              ▼
       ┌─────────────────────────────────────────────┐
       │ 4. CATEGORY & DYNAMIC POINT ENGINE          │
       │    Maps class -> Category & Base Points     │
       │    Sends UART Command (115200 baud)         │
       └──────────────────────┬──────────────────────┘
                              │
                              ▼
       ┌─────────────────────────────────────────────┐
       │ 5. ESP32-WROOM-32D + HIGH-TORQUE SERVO FLAP │
       │    • 60°  -> 🟢 BIODEGRADABLE BIN (Daily)   │
       │    • 120° -> 🔵 RECYCLABLE BIN (Every 3d)   │
       │    • 180° -> ⚫ NON-RECYCLABLE BIN          │
       │    • 240° -> 🔴 E-WASTE / HAZARDOUS BIN     │
       └──────────────────────┬──────────────────────┘
                              │
                              ▼
       ┌─────────────────────────────────────────────┐
       │ 6. DYNAMIC SESSION QR HANDSHAKE             │
       │    • Machine Kiosk generates 1-time QR      │
       │    • Student scans QR on Mobile App         │
       │    • EcoPoints + 1.5x Streak credited       │
       │    • Redeemable at Campus Canteens & Mess   │
       └─────────────────────────────────────────────┘
```

---

## 2. Dataset & AI Model Evaluation (`waste-classification-ai-main`)

The machine software builds directly upon the **TrashNet** benchmark repository (`waste-classification-ai-main`) and extends it to cover campus canteen sub-classes.

### 2.1 Dataset Verification & Stratified Split (Train / Validation / Test)
- **Dataset Source:** TrashNet resized variant (`512 × 384` RGB `.jpg` images, `0` corrupted files).
- **Total Images:** **2,527 images** across 6 core material classes.
- **Stratified Split Protocol (`seed = 42`):**
  - **Training Set (70%):** `1,768 images` (with `RandomResizedCrop(224)`, `RandomHorizontalFlip`, `RandomRotation(15°)`, `ColorJitter`, and ImageNet normalization).
  - **Validation Set (15%):** `379 images` (deterministic `Resize(224×224)` + ImageNet normalization).
  - **Test Set (15%):** `380 images` (held-out final evaluation).
- **Class Imbalance Mitigation:**
  - The dataset has an imbalance ratio of **`4.34×`** (`paper`: 594 images vs. `trash`: 137 images).
  - Mitigated during training using **inverse-frequency class-weighted cross-entropy loss**, boosting `trash` test recall to **95.0%**.

| TrashNet Class | Total Images | Train (70%) | Val (15%) | Test (15%) | ResNet50 Precision | ResNet50 Recall | ResNet50 F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cardboard** | 403 | 282 | 61 | 60 | 96.43% | 90.00% | **0.931** |
| **Paper** | 594 | 416 | 89 | 89 | 95.29% | 91.01% | **0.931** |
| **Glass** | 501 | 350 | 75 | 76 | 92.11% | 92.11% | **0.921** |
| **Metal** | 410 | 287 | 61 | 62 | 88.06% | 95.16% | **0.915** |
| **Plastic** | 482 | 337 | 72 | 73 | 95.45% | 86.30% | **0.906** |
| **Trash** | 137 | 96 | 21 | 20 | 63.33% | 95.00% | **0.760** |
| **Total / Macro** | **2,527** | **1,768** | **379** | **380** | **88.45%** | **91.60%** | **0.894 (91.05% Acc)** |

### 2.2 Multi-Model Benchmark Comparison
Four architectures were trained and benchmarked on an NVIDIA RTX 4050 (`seed = 42`):

| Model Architecture | Test Accuracy | Macro F1 | Weighted F1 | Parameters | Checkpoint Size | 1-Image Latency (Mean) | Deployment Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **ResNet50 (Selected)** | **91.05%** | **0.894** | **0.913** | 23.52M | 90.03 MB | **7.27 ms** | Primary RVM Kiosk Classifier |
| **MobileNetV2** | 82.11% | 0.805 | 0.823 | 2.23M | **8.75 MB** | **6.31 ms** | Lightweight Raspberry Pi / Edge Option |
| **EfficientNet-B0** | 80.53% | 0.788 | 0.808 | 4.02M | 15.61 MB | 12.59 ms | Evaluated Transfer Learning Baseline |
| **4-Block CNN (Scratch)** | 63.68% | 0.621 | 0.650 | 0.39M | 1.51 MB | 1.09 ms | From-Scratch Baseline |

### 2.3 Error Analysis & How the RVM Hardware Solves Model Ambiguities
In `reports/error_analysis.md` and `reports/gradcam_analysis.md`, only **34 / 380 test images (8.9%)** were misclassified by ResNet50. The top confusions were:
1. **`glass ↔ metal` (4 errors) & `plastic → metal` (3 errors):** Caused by glossy specular highlights on shiny plastic lids or glass jars looking metallic.
   - **Hardware Fix in RVM:** An **inductive proximity sensor (LJ12A3-4-Z/BX)** and **HX711 Load Cell** sit underneath the inspection chamber. Real metal triggers the inductive sensor, and real glass weighs `>120g` compared to `<30g` for plastic!
2. **`paper → trash` (4 errors) & low `trash` precision (63.3%):** Crumpled/soiled paper items can visually resemble generic `trash`.
   - **Economic Fix in RVM:** `Trash / Non-Recyclable` awards only **3 EcoPoints** (compared to 6–15 pts for clean recyclables), eliminating any incentive to game ambiguous items while still rewarding students for not littering.

---

## 3. Dynamic Pricing & Reward Points Valuation Matrix

Reward points are engineered using four factors:
1. **Local Recycler Scrap Value (₹/kg)** — Revenue generated for the campus sustainability fund.
2. **Lifecycle Carbon Offset ($\text{kg CO}_2\text{e}$ saved per item)**.
3. **Sanitary / Hazard Priority** — High bonus for keeping toxic batteries (`E-Waste`) and foul-smelling food waste (`Biodegradable`) properly segregated.
4. **AI Precision Calibration** — High-precision classes receive full material value; lower-precision `Trash` receives a baseline containment reward.

| Canteen Waste Item | YOLO Sub-Class | TrashNet Base Class | Target RVM Category | Servo Angle | Valid Weight Window | CO₂ Saved | Dynamic Points |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Spent Battery / Cell** | `battery` | `e_waste (ext)` | 🔴 **E_WASTE** | **240°** | 12g – 250g | 0.45 kg | **+25 pts** |
| **Aluminum Beverage Can** | `aluminum_can` | `metal` | 🔵 **RECYCLABLE** | **120°** | 10g – 120g | 0.22 kg | **+15 pts** |
| **Plastic PET Bottle** | `plastic_bottle` | `plastic` | 🔵 **RECYCLABLE** | **120°** | 8g – 85g | 0.14 kg | **+12 pts** |
| **Glass Bottle / Jar** | `glass_bottle` | `glass` | 🔵 **RECYCLABLE** | **120°** | 90g – 550g | 0.18 kg | **+10 pts** |
| **Cardboard Box / Carton** | `cardboard` | `cardboard` | 🔵 **RECYCLABLE** | **120°** | 15g – 400g | 0.11 kg | **+8 pts** |
| **Paper Cup / Sheet** | `paper` | `paper` | 🔵 **RECYCLABLE** | **120°** | 3g – 150g | 0.07 kg | **+6 pts** |
| **Banana Peel** | `banana_peel` | `organic (ext)` | 🟢 **BIODEGRADABLE** | **60°** | 15g – 350g | 0.09 kg | **+5 pts** |
| **Canteen Food Waste** | `food_waste` | `organic (ext)` | 🟢 **BIODEGRADABLE** | **60°** | 25g – 450g | 0.12 kg | **+5 pts** |
| **Chips Packet (MLP)** | `chips_packet` | `trash` | ⚫ **NON_RECYCLABLE** | **180°** | 1.5g – 90g | 0.04 kg | **+3 pts** |
| **Thermocol Plate (EPS)** | `thermocol` | `trash` | ⚫ **NON_RECYCLABLE** | **180°** | 2.0g – 90g | 0.05 kg | **+3 pts** |

*Note: Students maintaining a **5-Day Disposal Streak** automatically receive a **`1.5×` Streak Multiplier** on session checkout.*

---

## 4. Physical Hardware Bill of Materials (BOM) & Pinout Specification

### 4.1 Complete Bill of Materials (Per Canteen RVM Unit)

| Subsystem | Component | Specification / Model | Qty | Approx. Cost (INR) |
| :--- | :--- | :--- | :---: | :---: |
| **Compute & AI** | Edge AI Host / Kiosk Display | Raspberry Pi 5 (8GB) or Mini-PC + 10" Capacitive Touch Display | 1 | ₹8,500 |
| **Microcontroller** | Actuator Controller | ESP32-WROOM-32D DevKit V1 (Dual-Core, Wi-Fi/BLE, UART) | 1 | ₹450 |
| **Vision System** | Optical Inspection Camera | 1080p 60fps Wide-Angle USB Webcam + 5600K Diffused LED Ring | 1 | ₹1,400 |
| **Sorting Mechanism** | High-Torque Diverter Servo | DS3225 / MG996R 25kg·cm Digital Metal-Gear Servo (270° rotation) | 1 | ₹850 |
| **Singulator Belt** | Feeder & Conveyor Drive | NEMA-17 Stepper Motor + A4988 Driver + 40cm PVC Conveyor Belt | 1 | ₹1,650 |
| **Anti-Cheat Sensors** | Weight + 3D Depth + Drop | 5kg Load Cell + HX711 Amp, VL53L0X Laser ToF, IR Break-Beam | 1 set | ₹750 |
| **Bin Fill Sensors** | Ultrasonic Level Monitors | HC-SR04 Ultrasonic Distance Sensors (mounted on bin ceilings) | 4 | ₹320 |
| **Power & Enclosure** | Chassis & Power Supply | 12V/5V Dual SMPS + Acrylic/MDF Prototype Frame + 4 Color Sub-Bins | 1 | ₹2,800 |
| **Total Prototype Cost** | | **Hackathon MVP / Single Canteen Unit** | | **~₹16,720** *(or ~₹3,200 using laptop as kiosk)* |

### 4.2 ESP32 Wiring Pinout Table (`hardware_and_ai/esp32_rvm_controller.ino`)

| ESP32 GPIO Pin | Connected Hardware | Signal Type | Purpose |
| :--- | :--- | :--- | :--- |
| `GPIO 18` | DS3225 / MG996R Servo Signal | 50Hz PWM (`500–2500µs`) | Rotates chute diverter to `60°`, `120°`, `180°`, or `240°` |
| `GPIO 25` | A4988 `DIR` Pin | Digital Output | Sets conveyor belt forward/reverse direction |
| `GPIO 26` | A4988 `STEP` Pin | Digital Pulse | Advances conveyor belt to move one item under camera |
| `GPIO 4` | IR Break-Beam Receiver | Digital Input (`PULLUP`) | Confirms item physically dropped into target bin |
| `GPIO 32 / 33` | HX711 `DOUT` / `SCK` | Serial Digital | Measures item weight in grams (`0.1g` resolution) |
| `GPIO 21 / 22` | VL53L0X `SDA` / `SCL` | I²C Bus | Confirms 3D object height profile (blocks 2D phone photos) |
| `GPIO 12 / 14` | HC-SR04 `TRIG` / `ECHO` | Ultrasonic Pulse | Measures bin fill percentage (`0%–100%`) |

### 4.3 Machine Sizing & Collection Frequency Calculations
- **Average Hostel Canteen Footfall:** ~350 students/day disposing ~500 items/day.
- **Internal Bin Capacities (Total Footprint: `75cm W × 65cm D × 140cm H`):**
  1. **🟢 Biodegradable / Wet Bin (45 Liters):** Collected **Daily (Every 24 Hours at 8:00 PM)** to prevent odor/fermentation and feed the campus biogas/compost plant.
  2. **🔵 Recyclable Bin (120 Liters):** Collected **Every 2–3 Days** (fitted with an optional mechanical flap compactor for PET bottles & cans).
  3. **⚫ Non-Recyclable Bin (60 Liters):** Collected **Every 3 Days**.
  4. **🔴 E-Waste Safe Box (15 Liters, Fire-Retardant Lined):** Collected **Bi-Weekly**.

---

## 5. How to Run the Simulation & AI Software

### Option A: Run the Interactive Mobile & Kiosk Simulator (Zero-Install Browser App)
Open [`index.html`](file:///c:/Users/Archit/HTML/NSS_Sociothon/index.html) directly in any browser on your laptop or mobile phone:
1. **🤖 RVM Machine Tab:**
   - Tap any of the **10 Canteen Waste Items**, click **📥 Dump 5-Item Batch** to simulate collective disposal, or turn on **📷 Live Camera** / **🖼️ Upload Photo**.
   - Click **🪨 Test Fraud (Stone/2D Photo)** to demonstrate the 3-layer Anti-Cheat system.
   - Click **🏁 Done & Generate QR**, then tap **📲 Simulate Student Phone Scanning This QR**.
2. **🎓 Student App Tab:**
   - Scans the machine QR code, credits EcoPoints with the `1.5×` streak multiplier, unlocks **Canteen Vouchers**, displays the **Live Campus RVM Map**, **Dynamic Pricing Matrix**, and **Inter-Hostel Leaderboard**.
3. **📊 AI & Dataset Tab:**
   - Displays the **Part 4 Live Telemetry** counters, full **TrashNet 70/15/15 Train/Val/Test split evaluation**, **4-Model Benchmark**, and interactive **Grad-CAM & Confusion Matrix** figures from `waste-classification-ai-main`.
4. **⚙️ HW & Code Tab:**
   - Inspect the ESP32 wiring table, Python AI bridge code, and C++ microcontroller firmware directly inside the app.

### Option B: Run the Python ResNet50 + ESP32 Hardware Bridge
```bash
# 1. Activate environment with PyTorch & dependencies from waste-classification-ai-main
pip install -r waste-classification-ai-main/requirements.txt
pip install pyserial

# 2. Run the RVM AI + Microcontroller Serial Engine
python hardware_and_ai/rvm_ai_engine.py

# 3. Or launch the standalone Streamlit classifier demo
streamlit run waste-classification-ai-main/app/streamlit_app.py
```
