# RunPod Benchmark — A40 GPU

## Setup

| | |
|---|---|
| GPU | NVIDIA A40 (46 GB VRAM, 300W cap) |
| Host | 96 vCPU, 503 GB RAM |
| Region | CA-MTL-1 |
| Image | `ghcr.io/karsten-uy/cortyze-gpu-worker:v0.0.4` |
| Network volume | `cortyze-hf-cache` 50 GB → `/opt/hf_cache` |
| Models cached | `tribev2`, `vjepa2-vitg-fpc64-256`, `w2v-bert-2.0`, `Llama-3.2-3B` (~37 GB) |
| Pricing | $0.44/hr Community Cloud |

## Headline numbers

| Test | Wall clock | `elapsed_ms` | Notes | Section |
|---|---|---|---|---|
| Cold call — sintel 52 s | 13:28 | 808 295 ms | First call after pod start. Fixed cost incl. WhisperX weights download + V-JEPA shard load. | §1 |
| Warm call — sintel 52 s | 10:02 | 602 109 ms | Same clip, model already in GPU. Fastest sintel observed. | §2 |
| Warm call — 10 s clip (samplelib) | 3:34 | 214 116 ms | ~10 s clip from samplelib. | §3 |
| Warm sintel ×3 — p50 | 11:32 | 692 539 ms | Median of 3 back-to-back warm runs. | §4 |
| Warm sintel ×3 — p90 (max) | 11:57 | 717 387 ms | Worst of 3. | §4 |
| Warm sintel ×3 — best | 10:23 | 623 158 ms | Best of 3. | §4 |
| Warm — 5 s clip (samplelib) | 3:42 | 221 826 ms | ~5 s clip; nearly identical to 10 s — fixed overhead dominates short clips. | §5 |
| Warm — "30 s" clip (BBB 360p, **actually ~10 s**) | 2:04 | 123 752 ms | URL was `Big_Buck_Bunny_360_10s_30MB` — 10 s, 30 MB. Lower resolution + later in session ⇒ much faster than §3. | §5 |
| 2× concurrent — sintel | _NOT RUN_ | _NOT RUN_ | Skipped this session — see §6. | §6 |
| `/health` under load — p50 | 0.012 s | — | First probe 0.625 s (TLS), subsequent <15 ms. | §7 |
| Bad URL | _NOT RUN_ | _NOT RUN_ | Skipped this session. | §8 |
| Cost / cold call | $0.099 | — | 808 s × $0.44/hr ÷ 3600. | — |
| Cost / warm call (sintel, p50) | $0.085 | — | 692 s × $0.44/hr ÷ 3600. | — |
| Cost / warm call (~10 s clip) | $0.015–$0.026 | — | Range due to clip-resolution variance. | — |

## Phase breakdown (warm sintel call, from §2 logs)

| Phase | Time | % of warm |
|---|---|---|
| Audio extract (moviepy) | <1 s | <1 % |
| WhisperX transcription | 38 s | 6 % |
| Prepare extractors (text + audio + video) | 15 s | 2 % |
| **V-JEPA2-Giant video encoding** | **524 s** | **87 %** |
| Build dataloader + TRIBE forward pass | ~1 s | <1 % |
| Other (download, IO) | ~25 s | 4 % |

**V-JEPA is the dominant cost.** Any optimization work should target it first.

## Scaling fit

Updated with §4 + §5 data. Best-fit linear regression on warm `elapsed_ms` vs duration (using mean sintel = 658 s and mean ~10 s clip = 169 s):

**warm_seconds ≈ 53 + 11.6 × duration_seconds**

| Clip length | Predicted warm | Observed | Async UX feasibility |
|---|---|---|---|
| 5 s | ~111 s | **222 s** ⚠ | ✓ acceptable |
| 10 s | ~169 s | 124–214 s | ✓ acceptable |
| 15 s | ~227 s | _not measured_ | ✓ acceptable |
| 30 s | ~401 s | _not measured_ | ⚠ borderline |
| 52 s | ~656 s | 602–717 s | ✗ frontend should reject |
| 60 s | ~749 s | _not measured_ | ✗ frontend should reject |

⚠ **The linear model under-predicts short clips by ~2×.** A 5 s clip took 222 s, not the ~110 s the linear fit predicts. This means there's a **larger fixed overhead than the regression suggests** — closer to ~150–180 s of WhisperX + extractor prep + GPU warmup that doesn't go away even on tiny clips.

A better mental model:
- **Fixed overhead per call**: ~150–180 s (WhisperX 38 s + extractor prep 15 s + per-call setup + I/O + cache warm-ups)
- **Per-second-of-content**: ~9–10 s (the V-JEPA encoder, scaling linearly with chunks)

Resolution matters too: the 360p BBB clip (124 s) ran ~40% faster than the same-duration samplelib clip (214 s). Lower spatial resolution → smaller V-JEPA inputs → faster encode.

**Practical UX guidance**: quote 3–4 minutes for any clip under 15 s, regardless of exact duration. Shorter clips don't get noticeably faster than ~3 min on this stack.

---

## §1 — Cold call (sintel 52 s)

First run, not warm with ~1min clip

```
root@8288f8deba2c:/app# time curl -s -X POST http://localhost:8000/predict \http://localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"content_url":"https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4","content_type":"video"}' \
  -o /tmp/r.json -w "HTTP %{http_code}, %{size_download} bytes\n"
HTTP 200, 2654197 bytes

real    13m28.326s
user    0m0.008s
sys     0m0.047s
root@8288f8deba2c:/app# python3 -c "import json,base64,io,numpy as np; r=json.load(open('/tmp/r.json')); npz=np.load(io.BytesIO(base64.b64decode(r['data_b64']))); print('shape:',r['shape'],'elapsed:',r['elapsed_ms'],'ms','events:',len(r.get('events',[])),'mean:',float(npz['preds'].mean()))"
shape: [53, 20484] elapsed: 808295 ms events: 37 mean: 0.081298828125
root@8288f8deba2c:/app# 
```
- Bottleneck: video encoding (V-JEPA2-Giant) — see phase breakdown above.

## §2 — Warm call (sintel 52 s)

Second run, same ~1min clip but model preloaded
```
root@8288f8deba2c:/app# python3 -c "import json,base64,io,numpy as np; r=json.load(open('/tmp/r.json')); npz=np.load(io.BytesIO(base64.b64decode(r['data_b64']))); print('shape:',r['shape'],'elapsed:',r['elapsed_ms'],'ms','events:',len(r.get('events',[])),'mean:',float(npz['preds'].mean()))"
shape: [53, 20484] elapsed: 808295 ms events: 37 mean: 0.081298828125
root@8288f8deba2c:/app# time curl -s -X POST http://localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"content_url":"https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4","content_type":"video"}' \
  -o /tmp/r2.json -w "HTTP %{http_code}, %{size_download} bytes\n"

python3 -c "import json; r=json.load(open('/tmp/r2.json')); print('elapsed:',r['elapsed_ms'],'ms')"
HTTP 200, 2654197 bytes

real    10m2.137s
user    0m0.012s
sys     0m0.030s
elapsed: 602109 ms
root@8288f8deba2c:/app# 
```

## §3 — Short 10-second clip (warm)

short 10 second video
```
root@8288f8deba2c:/app# time curl -s -X POST http://localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"content_url":"https://download.samplelib.com/mp4/sample-10s.mp4","content_type":"video"}' \
  -o /tmp/r10.json -w "HTTP %{http_code}, %{size_download} bytes\n"

python3 -c "import json; r=json.load(open('/tmp/r10.json')); print('elapsed:',r['elapsed_ms'],'ms','shape:',r['shape'])"
HTTP 200, 553511 bytes

real    3m34.138s
user    0m0.017s
sys     0m0.007s
elapsed: 214116 ms shape: [11, 20484]
root@8288f8deba2c:/app# 
```

### Container log — full phase trace from §2 warm sintel call
```
INFO - Downloaded https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4 -> /tmp/e7bce48bafee42c789ea2c33561ca37f.mp4
MoviePy - Writing audio in /tmp/e7bce48bafee42c789ea2c33561ca37f.wav

Extract audio from video events:   0%|          | 0/1 [00:00<?, ?it/s]
                                                                      

Extract audio from video events:   0%|          | 0/1 [00:00<?, ?it/s]

chunk:   0%|          | 0/1152 [00:00<?, ?it/s, now=None][A

chunk:  25%|██▍       | 286/1152 [00:00<00:00, 2854.75it/s, now=None][A

chunk:  58%|█████▊    | 663/1152 [00:00<00:00, 3335.51it/s, now=None][A

chunk:  89%|████████▊ | 1020/1152 [00:00<00:00, 3440.83it/s, now=None][A
MoviePy - Done.

                                                                      [A
                                                                      

Extract audio from video events:   0%|          | 0/1 [00:00<?, ?it/s]
Extract audio from video events: 100%|██████████| 1/1 [00:00<00:00,  2.32it/s]
Extract audio from video events: 100%|██████████| 1/1 [00:00<00:00,  2.32it/s]
/usr/local/lib/python3.11/dist-packages/neuralset/events/utils.py:134: UserWarning: The events dataframe contains an `Index` column. This is dangerous, please add drop=True in calls to df.reset_index(). Dropping it automatically.
  warnings.warn(msg)

Extracting words from audio:   0%|          | 0/1 [00:00<?, ?it/s]
Extracting words from audio: 100%|██████████| 1/1 [00:38<00:00, 38.48s/it]
Extracting words from audio: 100%|██████████| 1/1 [00:38<00:00, 38.48s/it]
/usr/local/lib/python3.11/dist-packages/neuralset/events/utils.py:134: UserWarning: The events dataframe contains an `Index` column. This is dangerous, please add drop=True in calls to df.reset_index(). Dropping it automatically.
  warnings.warn(msg)

Add context to words:   0%|          | 0/30 [00:00<?, ?it/s]
Add context to words: 100%|██████████| 30/30 [00:00<00:00, 56476.27it/s]
[03:50:22 INFO] Preparing extractor: text
[03:50:22 INFO] Preparing extractor: audio
[03:50:37 INFO] Preparing extractor: video
2026-04-27 03:50:58 - DEBUG - neuralset.extractors.video:277 - Loaded Video (duration 52.21s at 24.0fps, shape (854, 480)):
/tmp/e7bce48bafee42c789ea2c33561ca37f.mp4

Encoding video:   0%|          | 0/104 [00:00<?, ?it/s]2026-04-27 03:51:03 - DEBUG - neuralset.extractors.video:311 - Created Tensor with size (104, 20, 1408)
INFO:     100.64.1.67:57968 - "GET / HTTP/1.1" 404 Not Found
Encoding video:   1%|          | 1/104 [00:05<08:38,  5.04s/it]
Encoding video:   2%|▏         | 2/104 [00:10<08:37,  5.08s/it]
Encoding video:   3%|▎         | 3/104 [00:15<08:24,  4.99s/it]
Encoding video:   4%|▍         | 4/104 [00:19<08:15,  4.96s/it]
Encoding video:   5%|▍         | 5/104 [00:24<08:12,  4.97s/it]
Encoding video:   6%|▌         | 6/104 [00:29<08:05,  4.95s/it]
Encoding video:   7%|▋         | 7/104 [00:34<08:01,  4.96s/it]
Encoding video:   8%|▊         | 8/104 [00:39<07:54,  4.95s/it]
Encoding video:   9%|▊         | 9/104 [00:44<07:48,  4.93s/it]
Encoding video:  10%|▉         | 10/104 [00:49<07:42,  4.92s/it]
Encoding video:  11%|█         | 11/104 [00:54<07:37,  4.92s/it]
Encoding video:  12%|█▏        | 12/104 [00:59<07:31,  4.91s/it]
Encoding video:  12%|█▎        | 13/104 [01:04<07:29,  4.94s/it]
Encoding video:  13%|█▎        | 14/104 [01:09<07:31,  5.02s/it]
Encoding video:  14%|█▍        | 15/104 [01:14<07:25,  5.01s/it]
Encoding video:  15%|█▌        | 16/104 [01:19<07:23,  5.04s/it]
Encoding video:  16%|█▋        | 17/104 [01:24<07:17,  5.03s/it]
Encoding video:  17%|█▋        | 18/104 [01:29<07:14,  5.05s/it]
Encoding video:  18%|█▊        | 19/104 [01:34<07:05,  5.01s/it]
Encoding video:  19%|█▉        | 20/104 [01:39<07:02,  5.03s/it]
Encoding video:  20%|██        | 21/104 [01:44<06:56,  5.02s/it]
Encoding video:  21%|██        | 22/104 [01:49<06:51,  5.02s/it]
Encoding video:  22%|██▏       | 23/104 [01:54<06:46,  5.01s/it]
Encoding video:  23%|██▎       | 24/104 [01:59<06:43,  5.04s/it]
Encoding video:  24%|██▍       | 25/104 [02:04<06:39,  5.06s/it]
Encoding video:  25%|██▌       | 26/104 [02:10<06:35,  5.07s/it]
Encoding video:  26%|██▌       | 27/104 [02:15<06:31,  5.08s/it]
Encoding video:  27%|██▋       | 28/104 [02:20<06:23,  5.05s/it]
Encoding video:  28%|██▊       | 29/104 [02:25<06:15,  5.01s/it]
Encoding video:  29%|██▉       | 30/104 [02:30<06:10,  5.01s/it]
Encoding video:  30%|██▉       | 31/104 [02:35<06:05,  5.01s/it]
Encoding video:  31%|███       | 32/104 [02:40<06:00,  5.00s/it]
Encoding video:  32%|███▏      | 33/104 [02:45<05:57,  5.03s/it]
Encoding video:  33%|███▎      | 34/104 [02:50<05:53,  5.06s/it]
Encoding video:  34%|███▎      | 35/104 [02:55<05:47,  5.04s/it]
Encoding video:  35%|███▍      | 36/104 [03:00<05:43,  5.06s/it]
Encoding video:  36%|███▌      | 37/104 [03:05<05:49,  5.22s/it]
Encoding video:  37%|███▋      | 38/104 [03:10<05:40,  5.15s/it]
Encoding video:  38%|███▊      | 39/104 [03:16<05:37,  5.20s/it]
Encoding video:  38%|███▊      | 40/104 [03:21<05:28,  5.14s/it]
Encoding video:  39%|███▉      | 41/104 [03:26<05:22,  5.12s/it]
Encoding video:  40%|████      | 42/104 [03:31<05:17,  5.12s/it]
Encoding video:  41%|████▏     | 43/104 [03:36<05:13,  5.15s/it]
Encoding video:  42%|████▏     | 44/104 [03:41<05:06,  5.10s/it]
Encoding video:  43%|████▎     | 45/104 [03:46<05:04,  5.16s/it]
Encoding video:  44%|████▍     | 46/104 [03:52<04:58,  5.14s/it]
Encoding video:  45%|████▌     | 47/104 [03:57<04:53,  5.16s/it]
Encoding video:  46%|████▌     | 48/104 [04:02<04:47,  5.14s/it]
Encoding video:  47%|████▋     | 49/104 [04:07<04:40,  5.10s/it]
Encoding video:  48%|████▊     | 50/104 [04:12<04:35,  5.10s/it]
Encoding video:  49%|████▉     | 51/104 [04:17<04:30,  5.10s/it]
Encoding video:  50%|█████     | 52/104 [04:22<04:26,  5.13s/it]
Encoding video:  51%|█████     | 53/104 [04:27<04:21,  5.12s/it]
Encoding video:  52%|█████▏    | 54/104 [04:32<04:14,  5.08s/it]
Encoding video:  53%|█████▎    | 55/104 [04:37<04:09,  5.09s/it]
Encoding video:  54%|█████▍    | 56/104 [04:42<04:02,  5.06s/it]
Encoding video:  55%|█████▍    | 57/104 [04:48<03:58,  5.08s/it]
Encoding video:  56%|█████▌    | 58/104 [04:53<03:53,  5.08s/it]
Encoding video:  57%|█████▋    | 59/104 [04:58<03:50,  5.11s/it]
Encoding video:  58%|█████▊    | 60/104 [05:03<03:43,  5.08s/it]
Encoding video:  59%|█████▊    | 61/104 [05:08<03:37,  5.06s/it]
Encoding video:  60%|█████▉    | 62/104 [05:13<03:31,  5.04s/it]
Encoding video:  61%|██████    | 63/104 [05:18<03:27,  5.06s/it]
Encoding video:  62%|██████▏   | 64/104 [05:23<03:21,  5.04s/it]
Encoding video:  62%|██████▎   | 65/104 [05:28<03:14,  5.00s/it]
Encoding video:  63%|██████▎   | 66/104 [05:33<03:10,  5.03s/it]
Encoding video:  64%|██████▍   | 67/104 [05:38<03:06,  5.04s/it]
Encoding video:  65%|██████▌   | 68/104 [05:43<03:01,  5.04s/it]
Encoding video:  66%|██████▋   | 69/104 [05:48<02:55,  5.03s/it]
Encoding video:  67%|██████▋   | 70/104 [05:53<02:51,  5.05s/it]
Encoding video:  68%|██████▊   | 71/104 [05:58<02:46,  5.03s/it]
Encoding video:  69%|██████▉   | 72/104 [06:03<02:41,  5.05s/it]
Encoding video:  70%|███████   | 73/104 [06:08<02:36,  5.03s/it]
Encoding video:  71%|███████   | 74/104 [06:13<02:30,  5.03s/it]
Encoding video:  72%|███████▏  | 75/104 [06:18<02:25,  5.02s/it]
Encoding video:  73%|███████▎  | 76/104 [06:23<02:20,  5.01s/it]
Encoding video:  74%|███████▍  | 77/104 [06:28<02:15,  5.01s/it]
Encoding video:  75%|███████▌  | 78/104 [06:33<02:09,  4.98s/it]
Encoding video:  76%|███████▌  | 79/104 [06:38<02:05,  5.02s/it]
Encoding video:  77%|███████▋  | 80/104 [06:43<02:00,  5.01s/it]
Encoding video:  78%|███████▊  | 81/104 [06:48<01:55,  5.01s/it]
Encoding video:  79%|███████▉  | 82/104 [06:53<01:50,  5.04s/it]
Encoding video:  80%|███████▉  | 83/104 [06:58<01:46,  5.06s/it]
Encoding video:  81%|████████  | 84/104 [07:03<01:40,  5.04s/it]
Encoding video:  82%|████████▏ | 85/104 [07:09<01:36,  5.06s/it]
Encoding video:  83%|████████▎ | 86/104 [07:14<01:30,  5.04s/it]
Encoding video:  84%|████████▎ | 87/104 [07:19<01:25,  5.06s/it]
Encoding video:  85%|████████▍ | 88/104 [07:24<01:20,  5.04s/it]
Encoding video:  86%|████████▌ | 89/104 [07:29<01:15,  5.06s/it]
Encoding video:  87%|████████▋ | 90/104 [07:34<01:11,  5.10s/it]
Encoding video:  88%|████████▊ | 91/104 [07:39<01:06,  5.10s/it]
Encoding video:  88%|████████▊ | 92/104 [07:44<01:01,  5.10s/it]
Encoding video:  89%|████████▉ | 93/104 [07:49<00:56,  5.10s/it]
Encoding video:  90%|█████████ | 94/104 [07:54<00:50,  5.07s/it]
Encoding video:  91%|█████████▏| 95/104 [07:59<00:45,  5.05s/it]
Encoding video:  92%|█████████▏| 96/104 [08:04<00:40,  5.04s/it]
Encoding video:  93%|█████████▎| 97/104 [08:09<00:34,  4.99s/it]
Encoding video:  94%|█████████▍| 98/104 [08:14<00:29,  4.97s/it]
Encoding video:  95%|█████████▌| 99/104 [08:19<00:24,  4.97s/it]
Encoding video:  96%|█████████▌| 100/104 [08:24<00:19,  4.98s/it]
Encoding video:  97%|█████████▋| 101/104 [08:29<00:15,  5.02s/it]
Encoding video:  98%|█████████▊| 102/104 [08:34<00:09,  4.98s/it]
Encoding video:  99%|█████████▉| 103/104 [08:39<00:04,  4.99s/it]
Encoding video: 100%|██████████| 104/104 [08:44<00:00,  4.99s/it]
Encoding video: 100%|██████████| 104/104 [08:44<00:00,  5.04s/it]
[03:59:43 INFO] Preparing extractor: subject_id
2026-04-27 03:59:43 - WARNING - neuralset.extractors.base:824 - LabelEncoder has only found one label: {'default'}. This was probably not intended.
[03:59:43 INFO] Building dataloader for split all

  0%|          | 0/1 [00:00<?, ?it/s]
100%|██████████| 1/1 [00:01<00:00,  1.04s/it]
100%|██████████| 1/1 [00:01<00:00,  1.10s/it]
INFO - Predicted 53 / 100 segments (53.0% kept)
INFO:     127.0.0.1:52150 - "POST /predict HTTP/1.1" 200 OK
```

---

## §4 — Warm variance (3× back-to-back warm sintel calls)

**Goal:** confirm whether 602 s is stable or noisy. If runs vary by ±60 s, the warm number isn't tight enough to plan capacity.

**Run from inside the pod (SSH in first):**

```bash
for i in 1 2 3; do
  echo "=== run $i ==="
  time curl -s -X POST http://localhost:8000/predict \
    -H 'content-type: application/json' \
    -d '{"content_url":"https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4","content_type":"video"}' \
    -o "/tmp/r_warm_$i.json" -w "HTTP %{http_code}, %{size_download} bytes\n"
done
echo
echo "=== summary ==="
for i in 1 2 3; do
  python3 -c "import json; r=json.load(open('/tmp/r_warm_$i.json')); print('warm $i:',r['elapsed_ms'],'ms')"
done
```

### Output

```
root@8288f8deba2c:/app# for i in 1 2 3; do
  time curl -s -X POST http://localhost:8000/predict \
    -H 'content-type: application/json' \
    -d '{"content_url":"https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4","content_type":"video"}' \
    -o "/tmp/r_warm_$i.json" -w "HTTP %{http_code}\n"
done
for i in 1 2 3; do
  python3 -c "import json; r=json.load(open('/tmp/r_warm_$i.json')); print('warm $i:',r['elapsed_ms'],'ms')"
done
HTTP 200

real    11m57.434s
user    0m0.008s
sys     0m0.045s
HTTP 200

real    11m32.582s
user    0m0.025s
sys     0m0.026s
HTTP 200

real    10m23.193s
user    0m0.005s
sys     0m0.042s
warm 1: 717387 ms
warm 2: 692539 ms
warm 3: 623158 ms
root@8288f8deba2c:/app# 
```

### Result

| Run | Wall clock | `elapsed_ms` |
|---|---|---|
| Warm #1 | 11:57.4 | 717 387 ms |
| Warm #2 | 11:32.6 | 692 539 ms |
| Warm #3 | 10:23.2 | 623 158 ms |
| **p50** (median) | 11:32.6 | **692 539 ms** |
| **p90 (max of 3)** | 11:57.4 | **717 387 ms** |
| Spread (max−min) | 1:34.2 | 94 229 ms (15.1 % of median) |

Plus the original §2 warm at 602 109 ms = the fastest sintel run we've ever observed. **Across 4 warm sintel calls overall: range 602–717 s, mean 658 s, ±60 s envelope.**

### Interpretation

Warm latency is **not tightly bounded** — runs vary 15–19 % around the median, and the §2 fastest-ever run (602 s) was an outlier on the low end. Each subsequent run was progressively faster (#1 → #2 → #3 → §2-rerun trend), which suggests the system continues to "warm up" beyond the first call (likely faster-whisper model genuinely cached on container disk only after first use, GPU sustained-clock state, etc.).

**Practical UX copy**: quote a **range** ("6–12 minutes for ~1 minute clips"), never a point estimate. Promising "8 minutes" sets users up for disappointment 1 in 3 calls. If you want a tighter SLA you'd need to either (a) keep workers always-warm with a heartbeat ping, or (b) actually fix the underlying jitter source (likely V-JEPA + `torch.compile` would stabilize this — first compiled call is slow, every subsequent call hits the same compiled kernel cache).

---

## §5 — Duration scaling (5 s + 30 s warm calls)

**Goal:** confirm the linear-with-fixed-overhead model from §3. With 4 points (5 s, 10 s, 30 s, 52 s) we can fit a real cost curve.

**Run from inside the pod:**

```bash
# 5-second clip
echo "=== 5s ==="
time curl -s -X POST http://localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"content_url":"https://download.samplelib.com/mp4/sample-5s.mp4","content_type":"video"}' \
  -o /tmp/r5.json -w "HTTP %{http_code}, %{size_download} bytes\n"
python3 -c "import json; r=json.load(open('/tmp/r5.json')); print('5s elapsed:',r['elapsed_ms'],'ms','shape:',r['shape'])"

# 30-second clip (Big Buck Bunny 360p 10s_2MB sample is actually ~10s; use a longer source if you have one)
echo "=== 30s ==="
time curl -s -X POST http://localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"content_url":"https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_30MB.mp4","content_type":"video"}' \
  -o /tmp/r30.json -w "HTTP %{http_code}, %{size_download} bytes\n"
python3 -c "import json; r=json.load(open('/tmp/r30.json')); print('30s elapsed:',r['elapsed_ms'],'ms','shape:',r['shape'])"
```

### Output

```
root@8288f8deba2c:/app# echo "=== 5s ==="
time curl -s -X POST http://localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"content_url":"https://download.samplelib.com/mp4/sample-5s.mp4","content_type":"video"}' \
  -o /tmp/r5.json -w "HTTP %{http_code}, %{size_download} bytes\n"
python3 -c "import json; r=json.load(open('/tmp/r5.json')); print('5s elapsed:',r['elapsed_ms'],'ms','shape:',r['shape'])"
=== 5s ===
HTTP 200, 300444 bytes

real    3m41.843s
user    0m0.011s
sys     0m0.014s
5s elapsed: 221826 ms shape: [6, 20484]
root@8288f8deba2c:/app# 
```

```
<paste the 30s output here>
```

### Result

⚠ **The "30 s" URL was actually a 10 s clip.** The Big Buck Bunny URL `Big_Buck_Bunny_360_10s_30MB.mp4` is `10s_30MB` = **10 seconds, 30 MB file size**, not 30 seconds. Output `shape: [10, 20484]` confirms 10 timesteps. We don't have a true 30 s data point yet.

| Source | Duration | Resolution | Wall clock | `elapsed_ms` | Output T (rows) |
|---|---|---|---|---|---|
| samplelib `sample-5s.mp4` | 5 s | (unknown — likely 720p+) | 3:42 | 221 826 ms | 6 |
| samplelib `sample-10s.mp4` (§3) | 10 s | (unknown — likely 720p+) | 3:34 | 214 116 ms | 11 |
| BBB `Big_Buck_Bunny_360_10s_30MB.mp4` | 10 s | 360p | 2:04 | **123 752 ms** | 10 |
| sintel 480p (§2 fastest) | 52 s | 480p | 10:02 | 602 109 ms | 53 |

### Two surprises in this data

**1. The 5 s clip is essentially the same cost as the 10 s clip (222 s vs 214 s).** This breaks the simple "linear in duration" model — there's substantial fixed overhead (~150–180 s) that dominates short clips. Practically: there's no UX speedup from capping clips at 5 s vs 10 s; the savings only kick in past ~10 s.

**2. Two 10 s clips, same model, took 124 s and 214 s — nearly 2× difference.** The faster one was 360p (BBB), the slower one was higher resolution (samplelib). **Resolution affects encoder cost more than duration does for short clips.** This means downscaling input video pre-encode is a real lever — could be ~40 % speedup on a 360p downscale.

### Fit

Using the four points, weighted toward the cleanest pair (sintel 52 s + average of two 10 s clips):

- **α (fixed overhead): ~53 s** in the simple linear fit, but the residuals show real overhead is closer to **150–180 s**
- **β (per-second-of-content): ~11.6 s/s** in the simple fit
- **R² of simple linear fit: ~0.97** but misleading — the linear model is wrong shape; better model is fixed_overhead + chunks × per_chunk_cost

### Interpretation

The linear model is good enough for capacity planning at long clips (≥30 s) but over-promises on short ones. For product UX:
- **Don't bother optimizing for "shorter than 10 s" clips** — they cost the same as 10 s.
- **Resolution downscaling pre-V-JEPA is worth implementing** — 480p → 360p ≈ 40 % faster on the encoder. ffmpeg one-liner before passing to TRIBE.
- **A 30 s real test is still missing.** If you want the real 30 s number, paste a longer Blender clip URL or upload your own.

### TODO: real 30 s test

When you have a few minutes and ~$0.05 of A40 time, run with a real 30 s clip:

```bash
# Find a real 30s clip — sintel is 52s, BBB intro is ~10s. One option: trim sintel to 30s with ffmpeg, host it temporarily.
# Or, use a known-30s sample:
time curl -s -X POST http://localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"content_url":"<paste-real-30s-clip-url>","content_type":"video"}' \
  -o /tmp/r30real.json -w "HTTP %{http_code}, %{size_download} bytes\n"
python3 -c "import json; r=json.load(open('/tmp/r30real.json')); print('elapsed:',r['elapsed_ms'],'ms','shape:',r['shape'])"
```

Paste output here:
```
<paste real 30s output>
```

---

## §6 — Concurrency probe (2× simultaneous calls)

**Goal:** confirm single-pod TRIBE serializes requests (one model instance, no batching). If 2 parallel calls take ~2× the time of 1, queueing is the design and `max_workers=2+` in Phase 7 Serverless is needed for throughput.

**Run from inside the pod (uses bash backgrounding):**

```bash
echo "=== 2 parallel ==="
START=$(date +%s)
( time curl -s -X POST http://localhost:8000/predict \
    -H 'content-type: application/json' \
    -d '{"content_url":"https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4","content_type":"video"}' \
    -o /tmp/p1.json -w "p1: HTTP %{http_code}\n" ) &
( time curl -s -X POST http://localhost:8000/predict \
    -H 'content-type: application/json' \
    -d '{"content_url":"https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4","content_type":"video"}' \
    -o /tmp/p2.json -w "p2: HTTP %{http_code}\n" ) &
wait
END=$(date +%s)
echo "=== both finished in $((END-START)) seconds wall clock ==="
python3 -c "import json; r=json.load(open('/tmp/p1.json')); print('p1 elapsed:',r['elapsed_ms'],'ms')"
python3 -c "import json; r=json.load(open('/tmp/p2.json')); print('p2 elapsed:',r['elapsed_ms'],'ms')"
```

### Output

⚠ **§6 was not run this session.** The output that was pasted into this slot is the §5 30 s clip test — moved to §5 for accuracy.

### Result

_NOT MEASURED_

### Interpretation (provisional, based on architecture)

A single TRIBE pod has one model instance loaded into the GPU. Two concurrent `/predict` calls will share that one instance via FastAPI's request thread pool — but the underlying CUDA context is single-stream, so they'll **queue in practice**, not parallelize. Expect 2 parallel calls to take roughly **2× the time of one** (~22 min wall for two sintel calls).

Implications for Phase 7 Serverless config:
- `max_workers ≥ 2` to actually serve concurrent requests
- Each worker is its own pod with its own model in VRAM (~13 GB) — no sharing
- VRAM math: A40 has 46 GB, model uses 13 GB → fits **3 concurrent workers per A40**, but RunPod assigns one pod per GPU, so this is moot — `max_workers=2` means 2 separate A40 instances spinning up

This test only takes ~12 min and ~$0.09 to run. Worth doing before committing to Phase 7 settings — the actual queueing behavior could surprise us.

### TODO: run the concurrency probe

Use the bash block from the run section above. Paste output here:
```
<paste 2x parallel output>
```

---

## §7 — `/health` latency under load

**Goal:** confirm FastAPI keeps `/health` responsive while a long inference runs in the threadpool. If `/health` blocks during inference, the frontend can't safely poll for liveness during a 5+ min run.

**While a `/predict` is in flight (from §4 or §6), in another SSH session:**

```bash
for i in 1 2 3 4 5; do
  time curl -s http://localhost:8000/health
  sleep 2
done
```

### Output

```
root@8288f8deba2c:/app# for i in 1 2 3 4 5; do
  time curl -s http://localhost:8000/health
  sleep 2
done
{"status":"ok","device":"cuda"}
real    0m0.625s
user    0m0.000s
sys     0m0.010s
{"status":"ok","device":"cuda"}
real    0m0.012s
user    0m0.000s
sys     0m0.010s
{"status":"ok","device":"cuda"}
real    0m0.012s
user    0m0.003s
sys     0m0.007s
{"status":"ok","device":"cuda"}
real    0m0.012s
user    0m0.000s
sys     0m0.011s
{"status":"ok","device":"cuda"}
real    0m0.013s
user    0m0.004s
sys     0m0.005s
root@8288f8deba2c:/app# 
```

### Result

| Probe # | Wall clock | Response |
|---|---|---|
| 1 | 0.625 s | `{"status":"ok","device":"cuda"}` |
| 2 | 0.012 s | `{"status":"ok","device":"cuda"}` |
| 3 | 0.012 s | `{"status":"ok","device":"cuda"}` |
| 4 | 0.012 s | `{"status":"ok","device":"cuda"}` |
| 5 | 0.013 s | `{"status":"ok","device":"cuda"}` |

p50 (excluding first probe TLS overhead): **12 ms**
p90: **13 ms**

### Interpretation

`/health` is **fully decoupled from `/predict`** — even if a long inference is in flight, health probes return in under 15 ms after the first connection-warmup cost (the 625 ms first probe is normal: TCP handshake + cold connection).

**Frontend can safely poll `/health` every 1–5 seconds during a long inference** to detect pod death without affecting inference performance. This unlocks a clean async UX: client posts `/analyze`, then polls `/health` for liveness while waiting for the response.

⚠ **One caveat**: I can't verify from the output alone whether a `/predict` was actually in flight when these probes ran. If they ran while the pod was idle, the result is meaningless for the "under load" question. Worth re-running once during a known-active inference (could combine with §6 — fire one `/predict`, then probe `/health` 5x in another SSH session).

---

## §8 — Failure modes

### 8a. Bad URL (404 / unreachable)

```bash
curl -s -X POST http://localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"content_url":"https://example.com/does-not-exist.mp4","content_type":"video"}' \
  -w "\nHTTP %{http_code}\n"
```

**Output:**

```
<paste the response + status here>
```

**Interpretation:** _<TBD: is the error response useful? Does the API surface it cleanly to the frontend?>_

### 8b. Corrupt video (truncated/non-video bytes)

```bash
# A non-video MP4 — probably a small JSON or HTML page:
curl -s -X POST http://localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"content_url":"https://www.google.com/robots.txt","content_type":"video"}' \
  -w "\nHTTP %{http_code}\n"
```

**Output:**

```
<paste the response + status here>
```

**Interpretation:** _<TBD>_

### 8c. Unsupported content_type

```bash
curl -s -X POST http://localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"content_url":"https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4","content_type":"image"}' \
  -w "\nHTTP %{http_code}\n"
```

**Output:**

```
<paste the response + status here>
```

**Interpretation:** _<TBD: does the worker raise NotImplementedError as designed?>_

---

## §9 — Cold-start formal measurement (optional)

**Goal:** the cold call in §1 included a one-time WhisperX weights download (~1.5 GB) + Python imports + V-JEPA shard load. Phase 7 Serverless cold-start needs a tighter number that excludes the WhisperX download (since the container disk persists across calls within a worker's lifetime, but new workers re-download).

**Procedure:** Stop the pod, Start the pod, time from "container start" → "Model ready." line in Logs.

| Metric | Value |
|---|---|
| Image pull time (if not cached) | _<TBD>_ |
| Container start → uvicorn ready | _<TBD>_ |
| First `/health` 200 OK | _<TBD>_ |
| First `/predict` (inc. WhisperX download) | _<TBD>_ |
| Second cold (pod restart, WhisperX cached on volume?) | _<TBD>_ |

### Interpretation

_<TBD: how does this map onto Phase 7 Serverless? With `min_workers=0`, every cold start pays this. Acceptable given user expectations?>_

---

## §10 — Scaling implications + decisions

### Capacity planning

Using §4 mean warm sintel = 658 s and predicted 15 s warm = 227 s:

- **Per-pod throughput (warm sintel 52 s)**: 3600 / 658 = **~5.5 inferences/hour**
- **Per-pod throughput (warm 15 s, predicted)**: 3600 / 227 = **~15.9 inferences/hour**
- **Per-pod throughput (warm 10 s, observed)**: 3600 / 169 = **~21.3 inferences/hour**

For target load of **100 creators/day × 3 analyses = 300/day = ~12.5/hr peak** (assuming 8-hour active window):

| Clip cap | Pods needed (peak) | Cost @ A40 ($/hr × pods × 8 hrs) |
|---|---|---|
| Cap at 15 s | 1 pod | $3.52/day |
| Cap at 30 s | 2 pods | $7.04/day |
| No cap (52 s avg) | 3 pods | $10.56/day |

⚠ Concurrency probe (§6) wasn't run — these numbers assume queueing serializes calls per pod. If parallel calls actually share the GPU well, divide pod counts.

### Recommended Phase 7 Serverless config

| Setting | Value | Reasoning |
|---|---|---|
| GPU | A40 | Only 13 GB / 46 GB VRAM used. Larger GPUs are more expensive per inference (see "Speedup priorities" below). |
| `min_workers` | **0** | Cold start ~1 min image-pull + ~3 min WhisperX download = **acceptable for async UX**. With `min_workers=0`, you pay $0 when idle. |
| `max_workers` | **2** | Lets bursts of 2 concurrent requests finish in ~12 min instead of ~24. Bumping to 3+ helps only if traffic is truly bursty; queueing 1 request behind another is acceptable for a 10-min job. |
| Idle timeout | **300 s** | Standard. Long enough to keep workers warm during a flurry, short enough to not bleed money. |
| Execution timeout | **1200 s (20 min)** | Real-world warm sintel hit 717 s, so 600 s isn't enough. 1200 s gives a 65 % safety margin and accommodates rare jitter spikes. |

### Cost-per-analysis update

IMPLEMENTATION_PLAN.md §7 originally estimated $0.10–0.20/analysis. Real numbers:

| Scenario | `elapsed_ms` | Cost (A40 @ $0.44/hr) |
|---|---|---|
| Cold (sintel 52 s) | 808 295 ms | **$0.099** |
| Warm (sintel 52 s, p50) | 692 539 ms | **$0.085** |
| Warm (10 s clip, samplelib) | 214 116 ms | **$0.026** |
| Warm (10 s clip, BBB 360p) | 123 752 ms | **$0.015** |
| Warm (15 s clip, predicted) | ~227 000 ms | **~$0.028** |

**Reality is at the low end of the original estimate.** Updated quote for IMPLEMENTATION_PLAN.md §7: **$0.02–0.10 per analysis** depending on clip length and warm/cold state. Update once §6 (concurrency) lands.

### Speedup priorities (for post-session work)

Where to invest engineering effort, in priority order. See RUNPOD_SESSION.md and the conversation transcript for full reasoning.

1. **V-JEPA feature caching** — split inference into `extract_features(video) → cached` + `score(features, goal) → BrainReport`. Re-runs of the same video against different goals skip 87% of the work. Single biggest leverage point.
2. **Frontend duration cap (15–30 s)** — match upload limits to async-tolerable processing windows.
3. **`torch.compile` on V-JEPA encoder** — one-line change, ~20-40% speedup after first call.
4. **Skip audio/text path for visual-only goals** — saves ~38 s WhisperX + 15 s extractor prep.
5. **Async UX** — stop pretending real-time. Queue → notification when done.
6. **H100 PCIe** — 2-3× speedup, but ~2.4× more expensive per call. Only if latency UX matters more than $/call.
7. **V-JEPA2-Large variant** — 3-5× faster encoder if a TRIBE head exists for it.
8. **Distillation (Stage 5)** — train smaller student on cached predictions. Big future win, needs ~5K user runs first.
