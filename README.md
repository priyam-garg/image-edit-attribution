# Image Edit Attribution

Identify **which editing pipeline produced an image**. A ResNet-18 classifier is trained to attribute an image to one of five editing sources, and Grad-CAM shows which regions drive each prediction.

## Editing pipelines (classes)

| Class | What it simulates | How it's generated |
|---|---|---|
| `original` | Unedited image | No change |
| `mobile_filter` | Phone/Instagram-style filter | Saturation ×1.25, contrast ×1.15, slight Gaussian blur |
| `photoshop` | Desktop editor retouch | Unsharp mask, brightness boost, vignette |
| `copy_move` | Copy-move forgery | A 25% patch is cropped, mirrored and pasted elsewhere |
| `gan_regen` | Generative / re-encoded image | JPEG (q=40) re-compression, desaturation, median filter |

## Pipeline

1. **Data** – Downloads [STL-10](https://cs.stanford.edu/~acoates/stl10/) (96×96 images) and exports it as PNGs.
2. **Synthetic dataset** – Applies each pipeline to source images:
   - Train: 600 images × 5 classes = 3,000
   - Test: 200 images × 5 classes = 1,000 (built from the STL-10 test split)
3. **Model** – ImageNet-pretrained ResNet-18 with a new 5-way head; inputs resized to 224×224.
4. **Training** – Adam (lr `1e-4`, weight decay `1e-5`), batch size 32, 8 epochs, cross-entropy loss.
5. **Evaluation** – Classification report and confusion matrix on the test set.
6. **Explainability** – Grad-CAM on `layer4[-1]` to visualize what the model attends to.
7. **Inference** – Upload any image and get its predicted editing source.

## Results

Test accuracy: **95%** (1,000 images)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| copy_move | 0.95 | 0.86 | 0.91 |
| gan_regen | 1.00 | 0.99 | 1.00 |
| mobile_filter | 0.98 | 0.96 | 0.97 |
| original | 0.86 | 0.91 | 0.88 |
| photoshop | 0.95 | 1.00 | 0.97 |

Confusion matrix (rows = true, columns = predicted; order: copy_move, gan_regen, mobile_filter, original, photoshop):

```
[[173   0   0  23   4]
 [  0 199   1   0   0]
 [  0   0 193   7   0]
 [  9   0   2 182   7]
 [  0   0   0   0 200]]
```

Most errors come from `copy_move` vs `original`: a mirrored patch from the same image leaves few global traces, so it's the hardest edit to spot.

## Getting started

The notebook was written for **Google Colab** with a GPU runtime.

1. Open `image_edit_attribution.ipynb` in Colab.
2. Set **Runtime → Change runtime type → GPU**.
3. Run the cells in order. The first cell installs dependencies, downloads STL-10 (~2.6 GB), builds the dataset, trains, evaluates and saves Grad-CAM results.
4. Use `predict_uploaded_image(model, test_ds.classes)` to classify your own images.

To run locally instead:

```bash
pip install -r requirements.txt
```

Then change `ROOT` (default `/content/attribution_project`) to a local path and skip the Colab-only cells (`google.colab.drive` / `files.upload`).

## Outputs

| Path | Contents |
|---|---|
| `attribution_project/resnet_epoch{N}.pth` | Checkpoint after each epoch |
| `attribution_project/final_model.pth` | Final weights plus class names |
| `gradcam_results/gradcam_*.png` | Grad-CAM overlays for 10 random test images |

Datasets and model weights are not committed to this repo; the notebook regenerates them.

## Limitations

- Edits are synthetic and parameterized, so the model may not generalize to real-world editing software or real GAN/diffusion outputs.
- Train and test use the same pipeline parameters; robustness to different edit strengths has not been tested.
- STL-10 images are small (96×96) and get upscaled to 224×224.
