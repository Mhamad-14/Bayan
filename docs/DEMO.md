# Bayan Service Demo

This demo shows the main end-to-end capabilities of the Bayan bilingual citizen-feedback service.

## 1. Start the service

From the project root:

```bash
conda activate bayan
make serve
```

The API will be available at:

```text
http://127.0.0.1:8000
```

## 2. Check service health

```bash
curl -s http://127.0.0.1:8000/health | python -m json.tool
```

This confirms that the Bayan API is running and ready to receive requests.

## 3. Arabic end-to-end analysis

The `/v1/analyse` endpoint combines topic classification, entity extraction, and similar-case retrieval in one request.

```bash
curl -s -X POST http://127.0.0.1:8000/v1/analyse \
-H 'Content-Type: application/json' \
-d '{"text":"ظهرت رسالة خطأ عند رفع المستند في خدمات المياه","k":3}' \
| python -m json.tool
```

### What this demonstrates

- Arabic citizen-feedback processing
- Topic classification
- Confidence scoring
- Named-entity extraction
- Semantic retrieval of similar historical cases
- Previous case resolutions for decision support

For this example, the classifier identifies the topic as:

```text
digital_services
```

The response also returns similar resolved cases related to the same type of issue.

## 4. Bilingual batch classification

Bayan also provides a batch-classification endpoint so multiple feedback items can be processed in one request.

```bash
curl -s -X POST http://127.0.0.1:8000/v1/classify:batch \
-H 'Content-Type: application/json' \
-d '{"texts":["ظهرت رسالة خطأ عند رفع المستند في خدمات المياه","I need clarification on the documents required for Maintenance Appointments"]}' \
| python -m json.tool
```

This demonstrates bilingual processing in a single API call.

Expected topic categories for these examples include:

```text
digital_services
licensing
```

## 5. Individual API capabilities

### Topic classification

```bash
curl -s -X POST http://127.0.0.1:8000/v1/classify \
-H 'Content-Type: application/json' \
-d '{"text":"ظهرت رسالة خطأ عند رفع المستند في خدمات المياه"}' \
| python -m json.tool
```

### Entity extraction

```bash
curl -s -X POST http://127.0.0.1:8000/v1/entities \
-H 'Content-Type: application/json' \
-d '{"text":"ظهرت رسالة خطأ عند رفع المستند في خدمات المياه"}' \
| python -m json.tool
```

### Similar-case retrieval

```bash
curl -s -X POST http://127.0.0.1:8000/v1/search \
-H 'Content-Type: application/json' \
-d '{"text":"ظهرت رسالة خطأ عند رفع المستند في خدمات المياه","k":3}' \
| python -m json.tool
```

## 6. Evaluation evidence

The repository includes evaluation evidence for:

- topic classification
- named-entity recognition
- semantic retrieval
- sliced evaluation by language, dialect, class, and text length
- behavioural testing
- model-card limitations
- manual error analysis
- optimization and serving benchmarks

See:

- `EVALUATION_REPORT.md`
- `BENCHMARKS.md`
- `DECISIONS.md`
- `model_cards/`

## 7. Serving and optimization

The serving classifier uses an optimized ONNX INT8 artifact.

The repository also retains an FP32 rollback path and documents the optimization decisions and paired quality comparison in:

```text
BENCHMARKS.md
DECISIONS.md
```

## 8. Run the automated tests

```bash
pytest -q
```

The test suite covers the main Bayan components and serving contracts.

## Demo summary

The Bayan service demonstrates an integrated bilingual NLP workflow:

```text
Citizen feedback
      ↓
Preprocessing
      ↓
Topic classification
      ↓
Entity extraction
      ↓
Semantic retrieval
      ↓
Similar resolved cases
```

The system exposes these capabilities through individual API endpoints and through the composite `/v1/analyse` endpoint for end-to-end use.
