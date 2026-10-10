# Separate PlantDoc Leaf Detector

PlantDoc TRAIN and TEST images, CSV rows, and Pascal VOC XML objects are prepared locally under this directory. The prepared detector split already exists at `dataset/`; do not rerun preparation unless you intentionally choose a different output directory.

The source annotations describe 29 categories such as `Tomato leaf bacterial spot`, `Apple Scab Leaf`, and `Blueberry leaf`. Their boxes enclose leaves, not the named disease or pest organism. The only pest-like phrase, `Tomato two spotted spider mites leaf`, occurs on two TRAIN leaf boxes and no TEST boxes; it is not an insect box label. Preparation therefore intentionally collapses source labels to one class, `leaf`. It does not make pest, healthy, non-crop, or four-class labels.

Preparation excludes invalid boxes and exact TRAIN duplicates of held-out TEST images. Its counts and exclusions are in `dataset/preparation_report.json`. The resulting splits contain 1,964 TRAIN, 354 validation, and 236 TEST images. All included images have leaf boxes; no non-crop negative images are supplied, so false-positive behavior on backgrounds is not measured.

## Train and evaluate

The separate Torchvision SSDLite trainer uses the already cached COCO initialization, freezes its pretrained backbone for a quicker transfer-learning run, and refuses to download model weights or overwrite existing outputs. By default it trains on an evenly spaced sample of 256 images and evaluates 32 images each from validation and held-out TEST; pass `--max-train-images 0 --max-validation-images 0 --max-test-images 0` to use every image (slower). Run from the project root:

```powershell
python -m quantum_ml.train_leaf_detector_v2 --epochs 3 --batch-size 4 --max-train-images 256 --max-validation-images 32 --max-test-images 32
```

The weights and sampled evaluation are saved under `models/plant_detector_v2/torch_ssdlite/`. The metrics file records the exact sample counts and the confidence threshold selected on validation. The trained run used 256 training images and evaluated 32 held-out TEST images at score threshold 0.35 and IoU 0.5: 37 true-positive boxes, 49 false positives, 20 missed boxes (precision 0.43, recall 0.65). This is a limited proof of concept, not field-ready performance; no background images were available to measure false positives on non-leaves. The runtime selects these weights when present; otherwise it retains the existing Ultralytics detector path. Restart the backend after training. The model can locate leaves in PlantDoc-like images only; disease classification requires a separately validated classifier, and pest detection is unsupported.
