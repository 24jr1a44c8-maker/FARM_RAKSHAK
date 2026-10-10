# FARM RAKSHAK

Hybrid classical computer vision and Qiskit machine learning prototype for camera-based crop monitoring. The intended workflow is **Detect → Locate → Quantify → Prioritize → Act**: inspect a live camera frame, associate valid detections with a selected farm zone, calculate the affected share, and surface zones for farmer inspection. The application does not prescribe pesticides.

## Current Readiness

The four-class classifier folders contain no images, and the older three-class model is rejected by the current four-class pipeline. A separate PlantDoc leaf detector has been trained under `models/plant_detector_v2/torch_ssdlite/`; it can return leaf boxes but cannot predict disease, healthy status, non-crop, or pests. On a held-out 32-image sample it achieved 0.43 precision and 0.65 recall at IoU 0.5 and score threshold 0.35; background false positives were not measurable because PlantDoc supplies no negative images. No accuracy or quantum advantage is claimed.

## Architecture

Camera frame → trained plant/leaf detector → crop each detected plant → MobileNetV2 CNN features → StandardScaler → 4-component PCA → Qiskit angle encoding → Qiskit Machine Learning VQC → confidence threshold → zone aggregation and risk map.

The detector locates plants/leaves; the CNN extracts crop features; the VQC performs four-class classification only when its matching trained artifacts exist. The separately trained PlantDoc SSDLite model can localize leaves, but it cannot identify disease, healthy status, non-crop backgrounds, or insect pests. When only this detector is ready, `/predict` reports leaf boxes and an explicit classification-unavailable status; those boxes never change health-zone counts. Development uses Qiskit's `StatevectorSampler` simulator, not quantum hardware. A simulator demonstrates a quantum circuit workflow but is not evidence of quantum advantage.

### PlantDoc Disease-Label Match (Scoped Fallback)

The original saved VQC is a three-class model (`Healthy`, `Pest`, `Disease`) and does not include `Non-Crop`; it is rejected by the four-class readiness gate. A direct held-out PlantDoc corn-blight crop check also showed that legacy model assigning 90.6% to `Pest`, so the app does not trust or connect it.

PlantDoc CSV boxes do contain explicit disease-category labels and generic leaf labels. A separate binary CNN + Qiskit VQC was trained over those boxes. Its classes mean **Disease-labeled leaf** and **Unspecified leaf**; generic leaf labels are not healthy labels, and the two spider-mite-labeled boxes are excluded. On the held-out TEST split it reached 0.47 accuracy and 0.47 macro F1, so it is marked research-only and is not enabled for live inference. The API keeps reporting classification unavailable rather than showing unreliable disease predictions. Even if enabled later, this limited classifier cannot identify pests, healthy plants, or non-crop images, and must not change zone health totals.

Prepare and train the isolated artifacts from PowerShell:

```powershell
python -m quantum_ml.prepare_plantdoc_disease
python -m quantum_ml.train_plantdoc_disease
```

The data preparation writes only to `dataset/plantdoc_disease_v2/`, and training writes only to `models/plantdoc_disease_v2/`. Both commands refuse to overwrite non-empty output directories. Training evaluates a separate validation split and PlantDoc's held-out TEST split. Better verified examples—especially genuinely healthy leaves that are not labeled as healthy merely by omission—are needed before a disease/no-disease classifier can be safely enabled.

## Environment Setup (Windows PowerShell)

From the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The installed PyTorch/torchvision and Qiskit versions need to be mutually compatible. CNN classifier training may download pretrained MobileNetV2 weights if they are not cached. The PlantDoc detector trainer uses only the already cached SSDLite COCO checkpoint and stops with an error if it is missing. Use localhost for camera permissions.

## Dataset Layout

Put real images in the matching folders. The labels are folder-based and must be correct:

```text
dataset/
  train/{healthy,pest,disease,non_crop}/
  validation/{healthy,pest,disease,non_crop}/
  test/{healthy,pest,disease,non_crop}/
```

The trainer requires at least one training image for every class and at least four training images total. Verify every image's label before putting it in a folder; the code uses folder names as ground truth and cannot verify them automatically. For meaningful evaluation, include representative validation and held-out test data for every class. Keep the test split separate from model fitting. The non-crop class should include people, soil, buildings, equipment, sky, and other backgrounds. Empty folders are not training data.

## Train the PlantDoc Leaf Detector

The existing PlantDoc files have already been prepared into a one-class leaf detector dataset under `models/plant_detector_v2/dataset`. Its CSV/XML classes are crop/disease names whose boxes enclose leaves. It has no background negatives, and its two spider-mite-named TRAIN boxes enclose leaves, not identified insects. It therefore supports leaf localization only, not insect pest detection or four-class classification.

The local Torchvision SSDLite COCO checkpoint is used only if already cached; the trainer will not fetch PlantDoc or model weights and will refuse to overwrite outputs. For a quicker transfer-learning run, it freezes the pretrained backbone and uses evenly sampled subsets by default. Run from the project root:

```powershell
python -m quantum_ml.train_leaf_detector_v2 --epochs 3 --batch-size 4 --max-train-images 256 --max-validation-images 32 --max-test-images 32
```

The new weights and sampled held-out metrics are saved separately under `models/plant_detector_v2/torch_ssdlite/`; the metrics record the exact sample sizes and validation-selected confidence threshold. Once created, the app uses them by default; restart the backend to load them. Because PlantDoc contains only positive leaf images, background false positives are not evaluated. Use `--max-train-images 0 --max-validation-images 0 --max-test-images 0` for full-split training/evaluation. The original detector file path remains available through `PLANT_DETECTOR_MODEL`.

## Train the Quantum Model

```powershell
python -m quantum_ml.train_quantum
```

The trainer extracts MobileNetV2 features, fits and saves the scaler/PCA, trains the VQC, and writes versioned metadata and SHA-256 artifact hashes after training succeeds. This ensures the classifier weights and preprocessing artifacts come from the same training run. If test images exist, it prints measured accuracy, per-class precision/recall/F1, confusion matrix, and a LogisticRegression baseline using the same reduced features. These values describe only the supplied dataset and are not generated when a test split is absent.

## Run the Application

From the project root, with the virtual environment active:

```powershell
uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000`. Allow camera access, select a zone, choose **Start Camera**, then **Start Continuous Scan**. On a phone, the rear-facing camera is preferred; use HTTPS or localhost to grant camera permission. Frames are sampled every 2.5 seconds and overlapping inference requests are refused. The frontend reports model/detector availability separately from a genuine no-plant result, and keeps the overlay aligned to the camera aspect ratio. Stop scanning and camera independently.

For a judge presentation, enable **Judge presentation demo (simulated labels)** in the live camera controls before scanning. It cycles Healthy, Pest, and Disease example labels over real leaf boxes from the camera detector; labels are explicitly marked as simulated, are not AI diagnoses, and are not written to alerts, scan history, or farm totals. No example labels are shown when the detector finds no leaf. Turn the option off for normal inference.

## Install on a Phone (Public HTTPS Deployment)

The project includes a Render Blueprint in `render.yaml`. It runs the FastAPI backend and frontend together, checks `/health`, and Render provides the HTTPS address required for phone camera access and installation. The configured `standard` instance is a paid plan; review Render's current pricing before creating it. The prototype currently has no sign-in, so anyone with the deployment URL can use its API, including zone reset endpoints.

To deploy:

1. Use your GitHub repository. The repository you provided is public, so the app source code will be visible to everyone after upload.
2. From PowerShell in the project root, initialize and push the app:

   ```powershell
   git init
   git add .
   git status --short
   git commit -m "Prepare FARM RAKSHAK for phone deployment"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/farm-rakshak.git
   git push -u origin main
   ```

   Check `git status` before committing. `.gitignore` excludes the virtual environment and large training datasets; keep the inference checkpoint `models/plant_detector_v2/torch_ssdlite/leaf_detector.pth` included.
3. In Render, choose **New + → Blueprint**, connect that GitHub repository, and deploy the detected `render.yaml`.
4. Open the HTTPS URL Render gives you on your phone. In Android Chrome, use the menu → **Install app** (or **Add to Home screen**). On iPhone, use Safari → **Share** → **Add to Home Screen**.

The app must stay deployed and online to scan live camera frames. Farm zone totals are in memory and may reset when the service restarts or redeploys.

## API

- `GET /health`: API and model readiness.
- `POST /predict`: predict one frame without changing zone counts; JSON body: `{"image":"data:image/jpeg;base64,..."}`.
- `POST /scan`: run inference and add valid detections to a zone; JSON body includes `zone_id` from 1 to 6.
- `GET /zones` and `GET /zones/{zone_id}`: current in-memory statistics.
- `POST /zones/{zone_id}/reset`: reset one zone.

Only confident Healthy, Pest, and Disease detections count toward zone totals. Non-Crop and uncertain results are excluded. Totals count per-frame detection events, not unique plants; repeated views of a plant can be counted again. Invalid/unready inference does not create a prediction or alter counts. Zone totals are in memory and reset when the backend restarts.

## Zone Risk

Affected percentage is `(pest + disease) / plants_scanned * 100`. The default prototype decision thresholds are configurable with `LOW_RISK_MAX_PERCENT` (default `5`) and `HIGH_RISK_MIN_PERCENT` (default `20`). Up to and including the low threshold is LOW; above it and below the high threshold is MEDIUM; at or above the high threshold is HIGH. Unscanned zones are labeled `NOT_SCANNED`. These are prototype thresholds, not scientifically validated agronomic guidance. Actions recommend inspection/monitoring only; consult qualified local agricultural guidance before treatment.

`PREDICTION_CONFIDENCE_THRESHOLD` configures the minimum VQC probability (default `0.55`). This is a prototype threshold and probabilities are not necessarily calibrated. Below it, inference is reported as uncertain and excluded from zone totals.

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Known Limitations

- The supplied classifier dataset folders are empty and the saved three-class model is rejected; the CNN + PCA + VQC classifier is not genuinely trained for the current four-class schema.
- The PlantDoc leaf detector is a limited proof of concept trained on 256 of 1,964 prepared training images. Its 32-image held-out sample had 49 false-positive boxes and 37 true-positive boxes at the validation-selected score threshold. Background false-positive rates are not measured, and the model must not be described as a pest detector.
- The disease/healthy/non-crop/pest classifier remains unavailable. No pest-insect boxes or explicit four-class image folders exist in PlantDoc.
- Current zone totals are process-local memory, not a durable farm database; restarting the backend clears them.
- The prototype counts each valid region in each sampled frame. It does not track plant identities across frames, so repeated views of the same plant may be counted more than once.
- Risk thresholds and confidence values require domain validation and representative calibration before field use.
- The VQC uses a local statevector simulator. There is no quantum-hardware run or measured quantum advantage claim.
