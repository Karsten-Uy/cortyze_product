# RunPod Session Checklist

Single-session checklist to: (1) deploy the GPU worker, (2) test the full pipeline end-to-end, (3) benchmark for scaling decisions, (4) tear down. Total expected spend: **~$5–7** + $3.50/mo for the volume. Total wall time: **~3 hours**.

Open this in a tab. Tick boxes as you go. Don't skip phases — phase 7 (serverless) depends on phase 6 (pod) finishing cleanly.

---

## Phase 0 — Pre-flight (free, ~5 min)

Before you spend a cent, confirm everything's ready locally:

- [ ] OrbStack running: `docker ps` returns without error
- [ ] `HF_TOKEN` env var set + gated `facebook/tribev2` accepted at <https://huggingface.co/facebook/tribev2>
- [ ] GitHub PAT with `write:packages` scope created at <https://github.com/settings/tokens>
- [ ] All local code committed (so you can `git diff` later if anything mysterious changes)
- [ ] Mock mode pipeline still works: `make api-run` + frontend → submit a video → see suggestions

```bash
# Quick sanity check from cortyze_product/
docker --version
echo $HF_TOKEN | head -c 10        # should print "hf_..." prefix
git status                          # should be clean
```

---

## Phase 1 — Build + push the GPU image (free, ~10 min)

The image is what RunPod will run. Build locally on your Mac (cross-compiles to linux/amd64).

```bash
cd /Users/kirby/Documents/cortyze/cortyze_product

# Log into GHCR (one-time):
echo $GITHUB_PAT | docker login ghcr.io -u $GITHUB_USER --password-stdin

# Build + push (~5 min):
GITHUB_USER=$GITHUB_USER make image-build
```

- [ ] Image built without error
- [ ] Push completed; final URL printed (e.g. `ghcr.io/yourname/cortyze-gpu-worker:latest`)
- [ ] Visible at <https://github.com/users/$GITHUB_USER/packages/container/cortyze-gpu-worker>

> **If `neuralset==0.0.2` fails to install during build:** see Troubleshooting §A. Fix locally, rebuild.

**Save for later:** the full image URL.

---

## Phase 2 — RunPod account (free, ~5 min)

- [ ] Sign up at <https://runpod.io>
- [ ] Add **$15** of credit (Settings → Billing). Gives ~$8 buffer over the expected ~$7 spend.
- [ ] Generate API key (Settings → API Keys → Create), save it
- [ ] (If pulling private GHCR image): Settings → Container Registry Auth → add GHCR creds with your PAT

**Save for later:** RunPod API key, your account ID.

---

## Phase 2.5 — (Optional but recommended) Cloudflare R2 setup (~15 min, $0)

**Skip this phase** if you only want to validate the GPU pipeline via public URLs (Phase 5 covers that path with sintel CDN).

**Do this phase** if you want to additionally test the **drag-drop upload flow end-to-end**: browser uploads → R2 → pod fetches from R2 → full pipeline. Required to demo the actual user experience.

R2 free tier: 10 GB storage, 1M Class A ops, 10M Class B ops, $0 egress. **Payment method required at signup** (no charges expected on free tier).

### 2.5.1 Sign up + buckets

- [ ] Sign up at <https://dash.cloudflare.com> (or use existing account)
- [ ] Click **R2** in left sidebar → add payment method (free tier, no charges expected)
- [ ] **Create bucket** → `cortyze-uploads`
- [ ] **Create bucket** → `cortyze-predictions`

### 2.5.2 API token with limited blast radius

- [ ] R2 → **Manage R2 API Tokens** → **Create API token**
- [ ] Permissions: **Object Read & Write**
- [ ] Specify buckets: `cortyze-uploads, cortyze-predictions` (NOT all buckets)
- [ ] TTL: 90 days
- [ ] Click Create — save the **Account ID**, **Access Key ID**, **Secret Access Key**

### 2.5.3 CORS on the uploads bucket (required for browser PUTs)

- [ ] Click `cortyze-uploads` → **Settings** → scroll to **CORS Policy** → **Add CORS policy**
- [ ] Paste:
  ```json
  [
    {
      "AllowedOrigins": ["http://localhost:3000"],
      "AllowedMethods": ["PUT", "GET", "HEAD"],
      "AllowedHeaders": ["*"],
      "ExposeHeaders": ["ETag"],
      "MaxAgeSeconds": 3600
    }
  ]
  ```
- [ ] Save

### 2.5.4 Lifecycle rules (keeps you under free tier even if you forget to clean up)

- [ ] `cortyze-uploads` → Settings → **Object Lifecycle Rules** → Add → "Delete after 7 days"
- [ ] `cortyze-predictions` → Settings → **Object Lifecycle Rules** → Add → "Delete after 30 days"

### 2.5.5 Wire credentials into Cortyze

```bash
# Secret in keychain:
security add-generic-password -U -s 'cortyze-r2-secret' -a $USER -w 'PASTE_R2_SECRET_KEY_HERE'
```

Update `cortyze_product/.env` (non-secrets only — Account ID + Access Key ID are fine in .env per Cloudflare's threat model):

```env
# Comment out the MinIO override so the API talks to R2 instead:
# S3_ENDPOINT_URL=http://localhost:9000

R2_ACCOUNT_ID=<your-cloudflare-account-id>
R2_ACCESS_KEY=<your-r2-access-key-id>
R2_BUCKET_UPLOADS=cortyze-uploads
R2_BUCKET_PREDICTIONS=cortyze-predictions
```

Update your `cortyze-secrets` shell function (in `~/.zshrc`) to also load `R2_SECRET_KEY`:

```bash
cortyze-secrets() {
  export GITHUB_PAT=$(security find-generic-password -s 'cortyze-github-pat' -w)
  export HF_TOKEN=$(security find-generic-password -s 'cortyze-hf-token' -w)
  export RUNPOD_API_KEY=$(security find-generic-password -s 'cortyze-runpod-api' -w)
  export R2_SECRET_KEY=$(security find-generic-password -s 'cortyze-r2-secret' -w)   # ← add this
  export DATABASE_URL=$(security find-generic-password -s 'cortyze-supabase-db' -w)
  echo "✓ Cortyze secrets loaded into shell env"
}
```

`source ~/.zshrc` to pick up the change.

### 2.5.6 Verify

```bash
cortyze-secrets
uv run python scripts/check_secrets.py
```

Should now show:
```
✓ R2/MinIO auth      R2 reachable, 2 bucket(s): cortyze-uploads, cortyze-predictions
✓ R2 named buckets   both named buckets reachable
```

If both go ✓, the upload flow is wired. The pod (deploying in Phase 4) will be able to fetch user uploads from R2's public-presigned URLs.

---

## Phase 3 — Network volume (~5 min, $0 today + $3.50/mo flat)

Persistent volume for HuggingFace weights — cached once, reused across every future pod. Without it, every cold pod start re-downloads ~20 GB.

**3.1 Create the volume:**
- [ ] RunPod dashboard → **Storage** → Network volumes → **Create**
- [ ] Name: `cortyze-hf-cache`
- [ ] Size: **50 GB**
- [ ] Region: **CA-MTL-1** (Montreal — A40 has High availability here; the volume locks region for all future pods that mount it)
- [ ] Cost shows ~$3.50/month flat

> **Why no CPU-pod weight pre-load?** CPU pods are unavailable in CA-MTL-1, and the cost difference between downloading on a CPU pod vs the A40 is ~$0.06 (CPU $0.06/hr vs A40 $0.44/hr × ~20 min). Not worth a second pod cycle. We download on the A40 directly in Phase 4.

---

## Phase 4 — Deploy the A40 GPU Pod + populate weights (~30 min, ~$0.20)

**Now the meter starts.** Plan to keep this pod alive for ~1.5–2 hours total (weight download + benchmarks), then stop.

### 4.1 Generate an SSH key (one-time, skip if you already have `~/.ssh/id_ed25519.pub`)

```bash
# On your Mac:
ls ~/.ssh/id_ed25519.pub 2>/dev/null || ssh-keygen -t ed25519 -C "cortyze-runpod" -f ~/.ssh/id_ed25519 -N ""
pbcopy < ~/.ssh/id_ed25519.pub   # public key now in clipboard
```

⚠️ Public key only (`.pub`). Never paste the private key.

### 4.2 Deploy the pod

- [ ] RunPod → **Pods → Deploy On-Demand**
- [ ] **GPU**: A40 48GB, **Community Cloud** (~$0.40/hr). If unavailable, fall back to A40 Secure (~$0.44/hr) or A100 40GB.
- [ ] Region locked to **CA-MTL-1** because the network volume is attached
- [ ] **Network volume**: select `cortyze-hf-cache`, **manually change the mount path from `/workspace` to `/opt/hf_cache`** (RunPod defaults this field to `/workspace`; the Dockerfile expects `/opt/hf_cache` and there's no symlink at runtime). If you forget, the worker re-downloads ~37 GB of weights to ephemeral container disk on first call and runs out of space mid-download.
- [ ] **Container image**: use a versioned tag, e.g. `ghcr.io/$GITHUB_USER/cortyze-gpu-worker:v0.0.4`. **Avoid `:latest`** — RunPod's image cache makes `:latest` updates unreliable; bumping the tag forces a clean pull every time.
- [ ] **Select Registry Auth**: pick the GHCR creds you saved in Phase 2 (otherwise the pull 404s if the GHCR package is private)
- [ ] **Container disk**: 30 GB (default 20 GB fills up fast once whisperx caches its own weights on first inference)
- [ ] **Expose HTTP ports**: `8000`
- [ ] **Expose TCP ports**: `22` (for SSH access — needed for weight download + debugging)
- [ ] **SSH public key**: paste the contents of `~/.ssh/id_ed25519.pub` (generate with `ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519` if you don't have one)
- [ ] **Container start command**: leave blank (use Dockerfile default)
- [ ] **Environment variables**: add **`HF_TOKEN`** = `<your-hf-token>`. **Required** — without it the worker 401s on gated `facebook/tribev2` and `meta-llama/Llama-3.2-3B` weights even though they're cached on the volume.
- [ ] Click Deploy

**While it boots (~30s):**
- [ ] Note the **Pod ID** from the dashboard
- [ ] Note the **public URL** — under "Connect" → form is `https://<pod-id>-8000.proxy.runpod.net`
- [ ] Note the **SSH command** — under "Connect" → form is `ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519`

**Save for later:** `POD_URL=https://<pod-id>-8000.proxy.runpod.net`

### 4.3 Pre-populate weights via SSH (~10 min, ~$0.07)

The first `/predict` call would otherwise trigger a ~37 GB model download mid-request and likely time out. Pre-populating into the network volume means every subsequent cold start uses cached weights.

The cortyze worker container will boot and crash-loop while the cache is empty (it can't import gated weights without HF_TOKEN until you set the env var, but even with the token, the model is too large to load before the first `/health` returns). Ignore the crashes — proceed.

```bash
# From your Mac, SSH into the pod (use the command from the Connect tab):
ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519

# Inside the pod:
export HF_TOKEN=hf_...           # paste your real token here (not the literal "hf_...")
export HF_HOME=/opt/hf_cache     # already set by Dockerfile, but be explicit

# `huggingface-cli` was renamed to `hf` in huggingface_hub 1.0+. Both the CLI
# and the underlying lib upgrade in one shot:
pip install --upgrade huggingface_hub

hf auth login --token $HF_TOKEN

# Four models, ~37 GB total, ~5 min on RunPod's bandwidth.
# NOTE the V-JEPA2 repo slug — it's `vjepa2-vitg-fpc64-256`, NOT `v-jepa2-...`.
hf download facebook/tribev2
hf download facebook/w2v-bert-2.0
hf download meta-llama/Llama-3.2-3B
hf download facebook/vjepa2-vitg-fpc64-256

# Verify:
du -sh /opt/hf_cache
ls /opt/hf_cache/hub | head
```

- [ ] All four downloads completed without error
- [ ] `du -sh /opt/hf_cache` shows ~37 GB
- [ ] `ls /opt/hf_cache/hub` lists 4 directories: `models--facebook--tribev2`, `models--facebook--vjepa2-vitg-fpc64-256`, `models--facebook--w2v-bert-2.0`, `models--meta-llama--Llama-3.2-3B`
- [ ] Force the worker to re-pick-up the cache: dashboard → pod → **Stop** → **Start**
- [ ] After ~30 s in Logs tab, see `Model ready.` and `Uvicorn running on http://0.0.0.0:8000`
- [ ] `curl https://<pod-id>-8000.proxy.runpod.net/health` returns `{"status":"ok","device":"cuda"}`

> **If you see "Cannot access gated repo":** accept terms at <https://huggingface.co/facebook/tribev2> and <https://huggingface.co/meta-llama/Llama-3.2-3B>, then re-run. Meta's LLaMA approval can take a few minutes.

> **If the V-JEPA2 download dies with "No space left on device":** the volume is too small. The 17 GB cached so far + the 20 GB of V-JEPA2 = 37 GB minimum. Resize to **50 GB** at Storage → your volume → Edit (volume must be detached from any pod). After resize, re-attach to the pod and resume the failed download.

> **A40 weight-download cost:** ~5 min × $0.44/hr ≈ **$0.04**. The volume now has the weights cached; future pod deploys skip this step.

---

## Phase 5 — Pipeline validation (~15 min, ~$0.10)

Confirm the worker actually serves real GPU inference and the cortyze API can drive it.

**5.1 Direct curl smoke test:**

⚠️ **Don't curl `https://<pod-id>-8000.proxy.runpod.net/predict` from your Mac for the first call.** RunPod's Cloudflare proxy enforces a **100-second timeout**. TRIBE v2's first inference on a real video takes 5–30 minutes (V-JEPA2-Giant encoding is GPU-bound but slow). Cloudflare returns HTTP 524 mid-inference; the worker keeps processing but you get no response, and a tqdm-on-closed-stderr crash can also fire when the cancelled request loses its file handles (the v0.0.4 image disables tqdm, but verify your image tag is current).

**Pattern: drive `/predict` from inside the pod** (localhost has no proxy, no timeout):

```bash
ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519

# Inside the pod:
curl -s http://localhost:8000/health
# Expect: {"status":"ok","device":"cuda"}

# First call — sintel ~52s clip. Cold-call wall-clock on A40: 5–30 min.
# (V-JEPA2-Giant + faster-whisper download on first run.)
time curl -s -X POST http://localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"content_url":"https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4","content_type":"video"}' \
  -o /tmp/r.json -w "HTTP %{http_code}, %{size_download} bytes\n"

# Inspect the response:
python3 -c "import json,base64,io,numpy as np; r=json.load(open('/tmp/r.json')); npz=np.load(io.BytesIO(base64.b64decode(r['data_b64']))); print('shape:',r['shape'],'elapsed:',r['elapsed_ms'],'ms','events:',len(r.get('events',[])),'mean:',float(npz['preds'].mean()))"
```

While you wait, in **another SSH session** watch GPU activity to confirm it's actually accelerating:
```bash
watch -n 2 nvidia-smi
# Expect GPU-Util at 80-100%, Memory ~20 GB, Power ~280-300W. If GPU-Util is 0%, encoder is on CPU and there's a config bug.
```

- [ ] `/health` returns `device:"cuda"`
- [ ] `/predict` returns HTTP 200 with shape `[~50, 20484]` (sintel is ~52s, so T should be in that range)
- [ ] First call cold-start: **5–30 min** on sintel (record actual number)
- [ ] Run it again — second call should be substantially faster (~30–90 s) because whisperx + V-JEPA caches are warm
- [ ] GPU-Util sat at 80–100% during inference (confirms GPU acceleration)

> **If first call exceeds 30 min**, check Logs for tracebacks. Common ones:
> - `FileNotFoundError: 'uvx'` → image missing uv. Rebuild from current Dockerfile.
> - `FileNotFoundError: 'ffmpeg'` → image missing ffmpeg. Rebuild from current Dockerfile.
> - `ValueError: I/O operation on closed file` (tqdm) → image was built before the v0.0.4 tqdm fix. Pull v0.0.4+.

> **Realistic latency**: TRIBE v2 was designed for batched offline research, not interactive inference. Even on A40, a 30-second clip is ~3–5 min warm. For the cortyze product UX you may need: shorter clip caps, async UX with progress, or a smaller V-JEPA variant. The benchmark numbers from this session feed those product decisions.

**5.2 Full API path through the local cortyze backend:**

Edit `cortyze_product/.env` to flip out of mock mode:
```env
INFERENCE_MODE=runpod
RUNPOD_POD_URL=https://<pod-id>-8000.proxy.runpod.net
RUNPOD_TIMEOUT_SECONDS=300
```

Then:
```bash
# Restart uvicorn
make api-run

# In another terminal, hit /analyze with a public URL:
curl -s -X POST http://localhost:8000/analyze \
  -H 'content-type: application/json' \
  -d '{"content_url":"https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4","content_type":"video","goal":"engagement"}' \
  | python -m json.tool | head -30
```

- [ ] `BrainReport` returned with non-zero region scores
- [ ] `region_scores.visual_cortex` is **within ±5 points of 64** (the CPU-fixture-derived golden value), confirming the GPU pipeline produces compatible output
- [ ] `moments` array is non-empty
- [ ] `brain_image_b64` is non-null
- [ ] `elapsed_ms` is in the 12,000–20,000 range

**5.3 Frontend visual check (Paste URL mode):**
- [ ] `cd ../cortyze_frontend && npm run dev`
- [ ] Open <http://localhost:3000>
- [ ] Use **Paste URL** mode
- [ ] Sintel URL pre-filled, click **Run BrainScore**
- [ ] See: brain heatmap, region cards with sparklines, suggestion cards on weak regions, dip/peak chips
- [ ] Click "▶ Jump to" on any chip — video seeks (sintel CDN supports this)

**5.4 Frontend upload flow (only if you completed Phase 2.5 with R2):**

This validates the full **drag-drop → R2 → pod → BrainReport** path. Skip if you didn't set up R2.

```bash
# Confirm S3_ENDPOINT_URL is commented out / unset in .env so the API uses R2, not MinIO:
grep -E "^S3_ENDPOINT_URL=" .env || echo "(not set — good)"

# Restart the API so it picks up the env change:
make api-run
```

- [ ] Frontend → switch to **Upload file** tab
- [ ] Drag a small video file (sintel mp4 you downloaded earlier works, or any short clip)
- [ ] Pick a goal, click **Run BrainScore**
- [ ] Watch progress: `Uploading...` → `Scanning brain...` → result
- [ ] BrainReport renders with real GPU-derived scores
- [ ] Cloudflare R2 dashboard → `cortyze-uploads` → Objects → see the file you just uploaded
- [ ] Cloudflare R2 dashboard → `cortyze-predictions` → see the `.npz` file with the request_id

**Acceptance for Goal 1 (full pipeline):** ✅ if all 5.1–5.3 checkboxes pass. Bonus ✅ if 5.4 also passes.

---

## Phase 6 — Pod-mode benchmark (~30 min, ~$0.30)

Now collect performance numbers.

**6.1 Cold-start metric (from pod logs):**
- [ ] RunPod → your pod → Logs tab
- [ ] Find the line `Loading TRIBE v2 onto cuda...` (container start)
- [ ] Find the line `Model ready.` (model loaded)
- [ ] Subtract: that's the cold-start time. Record it.

**6.2 Run the structured benchmark:**
```bash
cd /Users/kirby/Documents/cortyze/cortyze_product
RUNPOD_POD_URL=$POD_URL uv run python scripts/benchmark_runpod.py pod
```

This runs:
- 1 first inference (warm-ish since you already smoked it)
- 10 sequential warm calls (sintel)
- 1 call per clip across 3 clips of varying duration (sintel ~52s, big buck bunny ~10min, tears of steel ~12min)
- 2 then 4 parallel calls

It writes `runpod_benchmark.md` with everything.

- [ ] `runpod_benchmark.md` exists, all numbers populated
- [ ] Warm latency p50 is in 10–20s range
- [ ] Concurrency: 2 parallel finishes in roughly the same wall time as 1 sequential (queueing) — note this for your scaling write-up

**6.3 Peak VRAM (manual):**

Open a second terminal, SSH into the pod (RunPod → pod → Connect → SSH):
```bash
watch -n 0.5 nvidia-smi
```

While that's running, in your third terminal fire 2-3 inference calls in quick succession:
```bash
for i in 1 2 3; do
  curl -s -X POST "$POD_URL/predict" \
    -H 'content-type: application/json' \
    -d '{"content_url":"https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4","content_type":"video"}' > /dev/null &
done
wait
```

- [ ] Note peak `Memory-Usage` shown in nvidia-smi (form `XX MiB / 49140 MiB`)
- [ ] Add it to `runpod_benchmark.md` in the "manually captured" section

**6.4 STOP the Pod** before moving on. Dashboard → Stop. **Critical** — leaving it idle bleeds $0.40/hr.

- [ ] Pod status shows "Stopped"
- [ ] Volume `cortyze-hf-cache` still listed under Storage (it persists)

---

## Phase 7 — Convert to Serverless + benchmark (~30 min, ~$0.50)

**7.1 Create a Serverless endpoint:**
- [ ] RunPod → **Serverless → New Endpoint**
- [ ] Container image: same `ghcr.io/$GITHUB_USER/cortyze-gpu-worker:latest`
- [ ] **Container start command**: leave blank
- [ ] **Environment variable**: add `RUNPOD_SERVERLESS=1` ← **critical**, this flips handler.py into serverless-handler mode
- [ ] GPU: A40 48GB
- [ ] **Network volume**: mount `cortyze-hf-cache` at `/opt/hf_cache`
- [ ] Container disk: 30 GB
- [ ] **Active workers (min)**: `0`
- [ ] **Max workers**: `2`
- [ ] **Idle timeout**: `300` seconds
- [ ] **Execution timeout**: `300` seconds
- [ ] Click Create Endpoint

**Save for later:** the endpoint ID (visible on the endpoint page).

**7.2 Re-target the cortyze API:**

Edit `cortyze_product/.env`:
```env
INFERENCE_MODE=runpod
# Comment out the pod URL — having both is fine but pod takes precedence:
# RUNPOD_POD_URL=...
RUNPOD_API_KEY=<your-runpod-api-key>
RUNPOD_ENDPOINT_ID=<your-endpoint-id>
RUNPOD_TIMEOUT_SECONDS=300
```

Restart `make api-run`.

**7.3 First call (cold start):**
```bash
time curl -s -X POST http://localhost:8000/analyze \
  -H 'content-type: application/json' \
  -d '{"content_url":"https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4","content_type":"video","goal":"engagement"}' \
  | python -c "import sys,json; r=json.load(sys.stdin); print('overall:',r['overall_score'],'elapsed:',r['elapsed_ms'])"
```

- [ ] First call took **60–120s** (cold start for serverless includes container startup, not just model load)
- [ ] BrainReport returned correctly

**7.4 Run the benchmark in serverless mode (appends to existing report):**
```bash
RUNPOD_ENDPOINT_ID=<id> RUNPOD_API_KEY=<key> uv run python scripts/benchmark_runpod.py serverless --append
```

- [ ] `runpod_benchmark.md` updated with both Pod and Serverless sections side-by-side

**7.5 (Optional) Test cold-start tax:**

Wait 6+ minutes (idle timeout) without any requests. Then fire one. That's the realistic cold-start a real user will experience after low-traffic periods. Record it.

```bash
# After 6+ min wait:
time curl -s -X POST "http://localhost:8000/analyze" \
  -H 'content-type: application/json' \
  -d '{"content_url":"https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4","content_type":"video","goal":"engagement"}' > /dev/null
```

- [ ] Recorded the cold-start-after-idle number

---

## Phase 8 — Document, decide, tear down (~10 min, $0)

**8.1 Fill in the report's Scaling Implications section** — open `runpod_benchmark.md` and complete the bottom section:

- **min_workers / max_workers**: based on your concurrency results. If 4 parallel calls truly ran in parallel without queueing, max_workers=2 is enough; if they queued at 2, you may want max_workers=3.
- **idle_timeout**: longer = better cold-start UX, more cost. Recommend 300s for now.
- **GPU choice**: was peak VRAM under 40GB? A40 is sufficient. Over 40GB? Move to A100 80GB.
- **Per-analysis cost**: `(warm_p50_seconds * 0.40 / 3600)` for A40 = roughly $0.001-0.002/analysis at warm latency.

**8.2 Update IMPLEMENTATION_PLAN.md §7 cost estimate** with the real numbers (the existing $0.10–0.20/analysis was a guess; reality is probably 5-10x cheaper at warm).

- [ ] `runpod_benchmark.md` is complete (no `<TBD>` left)
- [ ] Cost estimate in IMPLEMENTATION_PLAN.md updated with real numbers

**8.3 Tear down:**
- [ ] Serverless endpoint: pause or delete (stays at $0/hr while idle anyway, but removing avoids any surprise)
- [ ] Pod: confirm it's STOPPED (or delete entirely)
- [ ] Network volume: keep (it's $3.50/mo flat and saves the 60-min repopulate next time)

**8.4 Reset local config to mock mode:**

Edit `.env`:
```env
INFERENCE_MODE=mock
# RUNPOD_POD_URL=
# RUNPOD_API_KEY=
# RUNPOD_ENDPOINT_ID=
```

Restart uvicorn → confirm mock mode works again. Save your API/endpoint values somewhere safe (e.g. 1Password) so re-activating later is fast.

- [ ] `.env` back in mock mode
- [ ] Mock-mode `/analyze` still returns BrainReport correctly
- [ ] Commit `runpod_benchmark.md` so the data survives

---

## Cost summary

| Phase | Time | Cost |
|---|---|---|
| 0 — Pre-flight | 5 min | $0 |
| 1 — Build + push image | 10 min | $0 |
| 2 — Account setup | 5 min | $0 (you load $15) |
| 3 — Network volume | 5 min | $0 today + $3.50/mo flat |
| 4 — Deploy A40 + populate weights | 30 min | ~$0.20 |
| 5 — Pipeline validation | 15 min | ~$0.10 |
| 6 — Pod benchmark | 30 min | ~$0.30 |
| 7 — Serverless | 30 min | ~$0.50 |
| 8 — Document + tear down | 10 min | $0 |
| **Total** | **~3 hr** | **~$1 + $3.50/mo for the volume** |

Buffer in your $15 deposit covers: misadventures, neuralset install issues, accidentally leaving the pod running for an extra hour, region-availability fallback to A40 Secure ($0.44/hr instead of $0.40/hr) or A100 40GB ($1.49/hr).

---

## Troubleshooting

### A. `neuralset==0.0.2` fails to install during `make image-build`

The package is in tribev2's deps. If pip can't resolve it, options in order of effort:
1. Try `pip install neuralset==0.0.2` directly — usually resolves from PyPI
2. If PyPI doesn't have it, vendor it: clone <https://github.com/facebookresearch/neuralset> and `pip install -e ./neuralset` BEFORE installing tribev2 in the Dockerfile
3. Pin tribev2 to a known-working commit if upstream broke

Fix in `gpu_worker/requirements.txt`, rebuild image.

### B. A40 Community unavailable in your region

Two paths:
- A40 Secure Cloud: ~$0.79/hr instead of $0.40/hr. Acceptable for this session, ~$0.40 extra spend.
- A100 40GB: ~$1.49/hr. Twice the cost but more available. Same VRAM headroom.

Update phase 4 GPU selection. No code changes needed.

### C. `/upload-url` flow doesn't work in Phase 5

Two cases:

**If you skipped Phase 2.5 (no R2):** This is **expected**. MinIO at `localhost:9000` isn't reachable from the RunPod pod. Use **Paste URL** mode with public URLs (sintel CDN) for Phase 5 validation. The drag-drop flow only works end-to-end with real R2.

**If you did Phase 2.5 (R2 configured):** Most likely cause is `S3_ENDPOINT_URL` is still set in `.env`, forcing the API to talk to MinIO instead of R2. Comment that line out, restart the API, retry. Second-likely cause: CORS not configured on `cortyze-uploads`; the browser PUT will fail with "blocked by CORS" in the dev tools console — fix per Phase 2.5.3.

### D. Pod boots but `/health` shows `device:"cpu"`

Pod doesn't have GPU access. Check pod's GPU assignment in the dashboard. May need to redeploy with explicit GPU selection. Don't bench on CPU — numbers won't be representative and the pod will hang on inference.

### E. First serverless call times out

Cold start can exceed your `RUNPOD_TIMEOUT_SECONDS=300` if the worker hasn't pulled the image yet (first request after endpoint creation). Bump to `RUNPOD_TIMEOUT_SECONDS=600` for the first call, lower it back after.

### F. Concurrency test shows queueing instead of parallelism

Pod mode runs one model instance — sequential by design. To test true parallelism you need either multiple pods or serverless with `max_workers=2+`. If the serverless concurrency=4 test ALSO shows queueing past worker count, that's RunPod's auto-scaling behavior. Spawning a second worker takes ~30-60s; it's intentional that bursts queue briefly.

### G. nvidia-smi shows VRAM > 45 GB on A40

Tight on the 48GB budget. Options:
- Reduce inference batch size in `gpu_worker/inference.py` (TRIBE v2 default is generous)
- Move to A100 40GB if you want more headroom (paradoxically slightly less VRAM than A40 but better memory bandwidth)
- A100 80GB if VRAM truly is the issue

This shouldn't happen for typical sintel-length clips; if it does, mention it in the benchmark report.

### H. Pod boots but worker crash-loops with `ModuleNotFoundError: No module named 'gpu_worker'` (or `services`, `core`)

The Dockerfile didn't put the package on Python's path, OR didn't COPY the supporting packages. Both fixed in current Dockerfile (`PYTHONPATH=/app` + `COPY services/`, `COPY core/`). If you see this error, your image predates the fix — rebuild with a bumped tag (`TAG=v0.0.X make image-build`) and redeploy.

### I. `FileNotFoundError: 'uvx'` or `'ffmpeg'` mid-inference

TRIBE v2 shells out to `uvx whisperx ...` for transcription and `ffmpeg` for audio extraction. The runpod/pytorch base image doesn't ship either. Both fixed in current Dockerfile (`apt-get install ffmpeg` + `pip install uv`). Rebuild with a bumped tag and redeploy.

### J. `ValueError: I/O operation on closed file.` (tqdm stack trace) mid-inference

Inference outlived its HTTP request (Cloudflare 100s proxy timeout). When FastAPI's threadpool task gets cancelled, its stderr handle dies, and tqdm crashes when it next tries to write a progress bar. **Fix is in v0.0.4+ image** (`TQDM_DISABLE=1` in Dockerfile + `verbose=False` in `inference.py`). Rebuild + redeploy if you see this.

The deeper architectural fix is to never drive `/predict` over the public proxy — see Phase 5.1 (SSH into the pod, curl `localhost:8000/predict` directly). Local mode has no Cloudflare timeout.

### K. HTTP 524 from RunPod proxy ("A timeout occurred")

Cloudflare gave up on your request at 100s. The pod is fine — the worker is still processing. Two options:

1. **Watch Logs tab** for the actual completion (`200 OK` line) and `elapsed_ms`. The first call always finishes server-side eventually.
2. **For real benchmarks**, drive `/predict` from inside the pod (SSH → `curl localhost:8000/predict`). Bypasses the proxy entirely.

Don't issue a second curl while one is in flight — it queues behind the first and makes things slower.

### L. `huggingface-cli: command not found`

The CLI was renamed to `hf` in `huggingface_hub` 1.0+. `pip install --upgrade huggingface_hub` then use `hf download <repo>` instead of `huggingface-cli download <repo>`. Same args.

### M. V-JEPA2 download dies with `No space left on device` even though the volume is "50 GB"

Two cases:
1. **Symlink confusion** — if `/opt/hf_cache` is a real directory (not a symlink to the network volume mount), downloads land on container disk. Verify with `ls -la /opt/hf_cache` (look for `lrwxrwxrwx`) or `df /opt/hf_cache` (should show `mfs#...runpod.net` filesystem). The cleanest fix is to set the network volume mount path to `/opt/hf_cache` directly in the pod config, not `/workspace`.
2. **Volume actually too small** — the four models total ~37 GB. `df -B1G /opt/hf_cache` shows the underlying MFS pool, not your quota. Trust the hf-xet error message — if it says "X MB free", that's your real quota. Resize at Storage → volume → Edit (must be detached from any pod first).

### N. RunPod proxy URL changes after Stop/Start

The pod URL form is `https://<pod-id>-8000.proxy.runpod.net`. The pod ID stays the same across Stop/Start, BUT may change if RunPod migrates the pod to different hardware (you'll see "Pod XXX migration completed successfully" notifications). When that happens:
- The Pod ID is preserved in the dashboard but the pod name shows "...-migration-migration".
- SSH host port and IP change (re-copy from Connect tab).
- HTTP proxy URL changes (re-copy from Connect tab).

Re-fetch URLs after every migration. Network volume contents persist.
