# API Examples

## Health check

Request: `GET /health`

```json
{"status": "ok", "model_loaded": "true"}
```

## Prediction request structure

`POST /predict` expects a JSON object that contains a `features` object with all 30 numerical fields. The full authoritative list is available from `GET /model-info`. One small excerpt is shown below; **do not send this excerpt by itself**, because the API requires all 30 features.

```json
{
  "features": {
    "mean radius": 14.0,
    "mean texture": 19.0,
    "mean perimeter": 90.0,
    "mean area": 600.0,
    "mean smoothness": 0.1
  }
}
```

The excerpt is intentionally incomplete to keep the documentation readable. For a valid request, copy all fields from a row of `load_breast_cancer(as_frame=True).data`, or use the complete feature list from `/model-info` and supply a numeric value for every field.

## Example success response shape

```json
{
  "predicted_class": "benign",
  "predicted_label": 1,
  "probabilities": {
    "malignant": 0.04,
    "benign": 0.96
  },
  "model": "StandardScaler + LogisticRegression(max_iter=3000)",
  "disclaimer": "Educational example only; this prediction must not guide medical decisions."
}
```

Probability values above are illustrative, not a guaranteed prediction. Actual values depend on the submitted feature measurements and trained model artifact.

## Input validation

- Missing or unknown feature names cause HTTP 422 with a helpful detail payload.
- Feature values must be numeric.
- JSON object key order does not matter; the API reorders features to match the training schema.
