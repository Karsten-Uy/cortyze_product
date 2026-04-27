# GPU Worker — RunPod deployment

This directory holds the code that runs **on RunPod GPUs**, not on your Mac. The cortyze API calls into it over HTTPS via [api/clients/runpod.py](../api/clients/runpod.py).

```
gpu_worker/
├── inference.py        # BrainPredictor — loads TRIBE v2 once, exposes predict()
├── handler.py          # FastAPI (Pod) + def handler(event) (Serverless), same model
├── requirements.txt    # Python deps installed into the Docker image
└── README.md           # ← you are here

../docker/
├── runpod.Dockerfile   # CUDA 12.1 + python 3.11 + tribev2; weights NOT baked in
└── build.sh            # docker buildx → push to GHCR

../api/clients/runpod.py    # MockRunPodClient (default) + RunPodClient (real)
```

The code is written. To go live you need to: **build the image**, **push it**, **set up RunPod**, and **switch the API to real-client mode**. Total cost to land all of this for the first time: ~**$5–15 of RunPod credit**.

---

## 1. Architecture

```
Browser → frontend:3000 → API:8000 → RunPodClient (urllib POST)
                                       │
                                       ▼ HTTPS
                          api.runpod.ai/v2/{endpoint_id}/runsync
                                       │
                                       ▼
                            ┌──────────────────────────┐
                            │ Serverless worker        │
                            │ (auto-scaled to 0 idle)  │
                            │                          │
                            │ handler(event)           │
                            │   → BrainPredictor       │
                            │     → TribeModel.predict │
                            │   → base64 npz           │
                            └──────────────────────────┘
                                       │
                                       ▼
                       /opt/hf_cache (network volume)
                       — model weights persist here
```

Two deployment modes share the same image and same handler.py:

| Mode | When | Cost shape |
|---|---|---|
| **Pod** | Iterating on `gpu_worker/` code; running smoke tests | Per-second, ~$0.40/hr (A40 Community) — stop the pod between sessions |
| **Serverless** | Production `/analyze` traffic | Per-second only while a request is running, $0/hr while idle |

You'll do `Pod → Serverless` once development converges. The image is the same.

---

## 2. Prerequisites

### Local

- **Docker / OrbStack / Colima** for image builds. On Apple Silicon, builds use `--platform linux/amd64` to cross-compile for RunPod's x86 GPUs. **Don't have one yet?** Install OrbStack — `brew install orbstack` then launch it. ~2 min, lighter than Docker Desktop.
- A **GHCR personal access token** with `write:packages` scope (free, GitHub Settings → Developer settings → PATs).

### RunPod

- Sign up at <https://runpod.io>, add ~$20 credit.
- API key from Settings → API Keys → Create.

---

## 3. One-time setup — pre-populate the network volume

This is the single biggest cost-saver: model weights live on a **persistent network volume**, not in the Docker image. Without it, every cold pod start re-downloads ~10 GB. With it, weights load in ~30 seconds from disk.

1. **RunPod dashboard → Storage → Network volumes → Create**
   - Name: `cortyze-hf-cache`
   - Size: 50 GB
   - Region: pick the same region you'll deploy to (US-EAST is widest GPU coverage)
   - Cost: ~$3.50/month flat, regardless of usage

2. **Spin up a one-time CPU pod to populate it.** RunPod dashboard → Pods → Deploy:
   - Template: any cheap CPU template (S0 or similar — ~$0.04/hr, this is just for downloading)
   - Network volume: mount `cortyze-hf-cache` at `/workspace`
   - Disk: defaults
   - Click **Deploy On-Demand**, wait ~60s for it to boot

3. **SSH or open the web terminal**, then:
   ```bash
   pip install huggingface_hub
   export HF_TOKEN=hf_...   # accept gated terms at https://huggingface.co/facebook/tribev2 first
   export HF_HOME=/workspace
   huggingface-cli download facebook/tribev2
   # Plus the three models tribev2 loads internally:
   huggingface-cli download facebook/v-jepa2-giant-fpc64-256
   huggingface-cli download facebook/w2v-bert-2.0
   huggingface-cli download meta-llama/Llama-3.2-3B
   ```
   This downloads ~10 GB total, ~30–60 min depending on bandwidth.

4. **Stop the pod.** The volume persists. Cost so far: ~$0.05.

You only do this once. Future GPU pods mount the same volume and skip the download.

---

## 4. Build the Docker image

From `cortyze_product/`:

```bash
# Log into GHCR
echo $GITHUB_PAT | docker login ghcr.io -u $GITHUB_USER --password-stdin

# Build + push (takes ~5 min on a fast Mac)
GITHUB_USER=yourname make image-build
```

Output:
```
Pushed: ghcr.io/yourname/cortyze-gpu-worker:latest
```

The image is ~3–5 GB. **No model weights inside** — those come from the network volume at runtime.

> Don't have local Docker? Alternative: GitHub Actions builds it for free. Add `.github/workflows/build-gpu-worker.yml` with a `docker/build-push-action` step triggered on push to main. Skipping that here — install OrbStack and use `make image-build` for now.

---

## 5. Deploy a Pod (for development)

RunPod dashboard → **Pods → Deploy**:

- **GPU:** A40 48 GB (Community Cloud) — recommended for cost. A100 40 GB also fine if you need it.
- **Template:** Custom → enter your image URL (`ghcr.io/yourname/cortyze-gpu-worker:latest`)
- **Container disk:** 30 GB
- **Volume:** mount `cortyze-hf-cache` at `/opt/hf_cache`
- **Expose HTTP ports:** 8000
- **Environment variables:** none required for pod mode
- Click **Deploy On-Demand**

When it boots (~30s), the **Connect** button shows a public URL like `https://<pod-id>-8000.proxy.runpod.net`. That's your worker.

### Smoke test from your Mac

```bash
POD_URL="https://<pod-id>-8000.proxy.runpod.net"
curl -s "$POD_URL/health"
# {"status":"ok","device":"cuda"}

curl -s -X POST "$POD_URL/predict" \
  -H 'content-type: application/json' \
  -d '{"content_url":"https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4","content_type":"video"}' \
  | python -c "import sys, json, base64, io, numpy as np; r = json.load(sys.stdin); npz = np.load(io.BytesIO(base64.b64decode(r['data_b64']))); print('shape:', r['shape'], 'elapsed:', r['elapsed_ms'], 'ms', 'mean:', npz['preds'].mean())"
```

You should see something like `shape: [53, 20484] elapsed: 12000 ms`.

### Stop the pod when done

RunPod dashboard → Pods → **Stop**. The pod and its container disk are gone; the network volume persists. **Always stop the pod between sessions.** Pods cost while running, even if idle.

---

## 6. Convert to Serverless (for production)

Once the pod path works:

RunPod dashboard → **Serverless → New Endpoint**:

- **Container image:** `ghcr.io/yourname/cortyze-gpu-worker:latest`
- **Container start command:** `python -u /app/gpu_worker/handler.py`
- **Environment variable:** `RUNPOD_SERVERLESS=1` ← important; switches handler.py into serverless mode
- **GPU:** A40 48 GB or A100 40 GB
- **Network volume:** mount `cortyze-hf-cache` at `/opt/hf_cache`
- **Active workers (min):** `0` ← scales to zero when idle = $0/hr
- **Max workers:** `2` (handles 2 concurrent users)
- **Idle timeout:** `300` seconds (keeps a worker warm for 5 min after last request)
- **Execution timeout:** `300` seconds

Click **Create Endpoint**. You'll get an `endpoint_id`. Cost while idle: $0.

---

## 7. Point the cortyze API at the deployed endpoint

In `cortyze_product/.env`:

```env
RUNPOD_API_KEY=<your-api-key>
RUNPOD_ENDPOINT_ID=<the-endpoint-id-from-step-6>
```

That's it. The `get_client()` selector in [api/clients/runpod.py](../api/clients/runpod.py) auto-detects both env vars and switches from `MockRunPodClient` to `RunPodClient`. No code changes.

Restart uvicorn:
```bash
make api-run
```

Now `/analyze` runs end-to-end: frontend uploads → R2/MinIO → API → RunPod GPU → real `(T, 20484)` → atlas → scoring → BrainReport.

---

## 8. Cost expectations

| Activity | Cost |
|---|---|
| Network volume (50 GB, monthly flat) | $3.50 |
| First-time weight download (~60 min CPU pod) | ~$0.05 |
| Image build / push | $0 (local Docker) or $0 (GitHub Actions) |
| Pod dev session, A40, 1 hour | ~$0.40 |
| Single `/analyze` call on Serverless A40 (~12s) | ~$0.001 |
| Stage 3 launch traffic: 100 analyses/day | ~$0.10/day = $3/mo |

**Realistic Stage 1.2 total to land RunPod integration:** $10–25, mostly the network volume's first month + a few hours of pod time during bring-up.

### The five biggest money-burners (avoid these)

1. **Building images on a running pod.** Build locally (~5 min, free) or in CI. Building on a paid pod burns ~$0.40/hr while you wait.
2. **Re-downloading weights on every cold start.** That's what the network volume prevents — never put HF weight downloads in the Dockerfile.
3. **Leaving pods running.** RunPod doesn't stop them automatically. `make pod-status` (after you record the pod ID) or just check the dashboard before walking away.
4. **Bigger GPUs than needed.** TRIBE v2 inference fits in 40 GB. A100 80GB and H100 are 2–3× the cost for zero speedup.
5. **Iterating on RunPod when you could iterate locally.** The mock client serves real BrainReports without GPU. Push to RunPod only for things that need actual inference.

---

## 9. Troubleshooting

**Pod boots but `/health` returns CPU instead of cuda**
The image isn't seeing the GPU. Check the pod has GPUs assigned (RunPod dashboard → pod details). May need to redeploy with explicit GPU selection.

**`predict` hangs on first call**
Model is loading from network volume (~30 sec) or downloading missing weights (much longer). Check pod logs: `Loading TRIBE v2 onto cuda...`. If it says `downloading from huggingface.co/...`, your network volume is missing files — repeat step 3 of section 3.

**`neuralset==0.0.2 not found`**
Listed as a dep in tribev2's pyproject. If pip can't find it, vendor manually: clone the repo and `pip install -e .` in the Dockerfile. Check tribev2 issues for current install instructions.

**RunPod returns `status=FAILED`**
Check the worker logs in the RunPod dashboard. Most common: out-of-memory (OOM) on the GPU. Try a larger GPU tier (A100 80GB) or check whether the input video is unusually long.

**Cold start takes >60s on Serverless**
That's expected with TRIBE v2. To eliminate: set `min_workers=1` (always-on) — costs ~$0.40/hr but eliminates cold starts entirely. Not worth it until you have real traffic.

**Mock mode broke after I set `RUNPOD_*` env vars**
That's the trigger to switch to real client. Unset both env vars and restart uvicorn to go back to mock mode.

---

## 10. What's next (Stage 1.2 → Stage 2)

After this lands, the remaining items are:

- **30-clip calibration** — run TRIBE v2 against ~30 reference clips on the deployed endpoint, recompute `core/scoring/calibration.json` from cross-clip statistics. Replaces the single-clip placeholder. ~$0.30 of inference cost.
- **Real R2 + Supabase** — swap MinIO for production R2 buckets and Supabase Postgres. Same env-var schema (per [README §3](../README.md)), just point at real services.
- **Stage 2 suggestion engine** — `services/suggestions/` calling Claude with the BrainReport + reference ad library matches. Per [IMPLEMENTATION_PLAN.md §6.7](../IMPLEMENTATION_PLAN.md).
