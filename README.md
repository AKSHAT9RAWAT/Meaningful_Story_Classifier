# Tag Meaningfulness Classifier

This project classifies tags as meaningful (1) or not (0) for event/activity/emotion detection, using cluster-level labels and classic ML (no transformers).

## Project Structure

- `src/` - Source code modules
- `models/` - Trained model files
- `tags/` - Tag data files
- `final_groundtruth.json` - Cluster-level tags and labels

## Setup

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Training

Train the model using cluster-level tags and labels:

```bash
python src/train_model.py --cluster_json final_groundtruth.json --model_path models/tag_classifier.pkl
```

## Inference

Run inference on individual image tags:

```bash
python src/inference.py --model_path models/tag_classifier.pkl --individual_json tags/full_individual_tags.json --output_path inference_results.json
```

## Customization

- You can add your own good/bad tag lists in `train_model.py` for improved feature engineering.
- The model uses tag frequency, length, and list membership as features.

## Notes

- No transformers or deep learning models are used.
- The pipeline is modular and easy to extend.
