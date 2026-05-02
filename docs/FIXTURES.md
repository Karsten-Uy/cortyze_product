# Generating Brain Prediction Fixtures

A fixture is a `(T, 20484)` `.npy` file containing real TRIBE v2 cortical
predictions for a piece of content. They live in [tests/fixtures/](tests/fixtures/)
and serve two purposes:

1. **Power mock mode** (`INFERENCE_MODE=mock`) so the API returns realistic
   brain scores without hitting RunPod.
2. **Anchor calibration** ([scripts/calibrate_from_fixture.py](scripts/calibrate_from_fixture.py))
   so the 0–100 region scores reflect "vs. our reference library" instead of
   "vs. one specific clip".

You run these scripts on the M3 8GB Mac in CPU + bf16 mode. Each fixture takes
~30–90 minutes wall time depending on inputs. Plan to kick them off and walk
away.

---

## Prerequisites (one-time)

```bash
brew install ffmpeg                       # video/audio assembly + muxing
uv tool install --python 3.11 whisperx    # CLI binary for audio transcription
uv tool install yt-dlp                    # only if grabbing from YouTube
```

Add `setopt interactivecomments` to `~/.zshrc` so `#` works as a comment
when pasting multi-line commands. Otherwise zsh prints `command not found: #`
on every comment line.

You need an HF token with access to two gated repos. Both will fail at
import time without this:
- [facebook/tribev2](https://huggingface.co/facebook/tribev2)
- [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B)

The token lives in macOS Keychain under the name `cortyze-hf-token`. The
`cortyze-secrets` shell function exports it as `HF_TOKEN`. Check first:

```bash
cortyze-secrets
echo "${HF_TOKEN:0:6}..."   # should print "hf_..."
```

If it's empty, store one once with:

```bash
security add-generic-password -s 'cortyze-hf-token' -a $USER -w 'hf_xxxxx'
```

---

## Two scripts, two content shapes

| Script | Input | Output stem |
|---|---|---|
| [scripts/build_fixture.py](scripts/build_fixture.py) | A video URL (mp4) | `golden_pred_video_<urlhash>_T<n>.npy` |
| [scripts/build_post_fixture.py](scripts/build_post_fixture.py) | An image URL + optional audio URL + optional caption | `golden_pred_post_<contenthash>_T<n>.npy` |

Both default output stems include a content-derived hash so unique inputs
write to unique files — re-runs with new content **will not overwrite** older
fixtures. Re-running with the **same** input errors with "already exists";
pass `--force` to override.

The legacy fixtures `golden_pred_sintel_T53.npy` and `golden_pred_post_T6.npy`
predate the hash naming and are kept around for backward compatibility.

---

## Workflow: video fixture from a YouTube ad

```bash
cd /Users/kirby/Documents/cortyze/cortyze_product
cortyze-secrets

# 1. Download MP4 (yt-dlp picks the best mp4 + m4a streams and merges them)
yt-dlp -f "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]" \
  -o "/tmp/myad.mp4" \
  "https://www.youtube.com/watch?v=YOURVIDEOID"

# 2. Optional but recommended: trim to 60s if the source is longer.
#    Saves ~60% wall time and keeps RAM in check on M3 8GB.
ffprobe /tmp/myad.mp4 2>&1 | grep Duration
ffmpeg -i /tmp/myad.mp4 -t 60 -c copy /tmp/myad_60s.mp4

# 3. Upload to R2 + get a 24h presigned URL
VIDEO_URL=$(uv run python scripts/upload_to_r2.py /tmp/myad_60s.mp4)
echo "$VIDEO_URL"

# 4. Run TRIBE
/Users/kirby/Documents/cortyze/tribev2/.venv/bin/python \
  scripts/build_fixture.py \
  --video-url "$VIDEO_URL"
```

Wall time: ~30–90 min depending on video length and whether the model weights
are already cached in `~/.cache/huggingface/`.

---

## Workflow: post fixture from image + caption (no audio)

```bash
cd /Users/kirby/Documents/cortyze/cortyze_product
cortyze-secrets

/Users/kirby/Documents/cortyze/tribev2/.venv/bin/python \
  scripts/build_post_fixture.py \
  --image-url 'https://images.unsplash.com/photo-XXXX?w=1200' \
  --caption 'Your caption text — drives the language regions.'
```

Wall time: ~30–50 min.

The image URL must be directly downloadable (Unsplash, your own R2 bucket,
etc). Instagram / Facebook CDN URLs work but expire within hours — copy the
image right before running.

---

## Workflow: post fixture from image + audio + caption (full)

This is the highest-fidelity post fixture: image, voiceover, and caption all
contribute. Wall time: ~50–90 min (WhisperX over the audio is the heaviest
single step).

```bash
cd /Users/kirby/Documents/cortyze/cortyze_product
cortyze-secrets

# 1. Record / generate / locate an audio file (15–30s of clear speech is ideal)
#    Easy option: macOS built-in TTS
say -v Samantha -o /tmp/voice.aiff "Your caption read aloud, exactly."
ffmpeg -i /tmp/voice.aiff -c:a aac -b:a 128k /tmp/voice.m4a

# 2. Upload audio to R2
AUDIO_URL=$(uv run python scripts/upload_to_r2.py /tmp/voice.m4a)
echo "$AUDIO_URL"

# 3. Run
/Users/kirby/Documents/cortyze/tribev2/.venv/bin/python \
  scripts/build_post_fixture.py \
  --image-url 'https://...' \
  --audio-url "$AUDIO_URL" \
  --caption 'Your caption text.'
```

If the image is on a CDN with a private bucket (e.g. an Instagram URL), you
can host it yourself first the same way:

```bash
curl -o /tmp/post.jpg 'https://instagram-cdn-url.jpg'
IMAGE_URL=$(uv run python scripts/upload_to_r2.py /tmp/post.jpg)
```

---

## After every fixture build

Whenever you add a new fixture (or replace an old one), re-pool calibration
so the score distribution accounts for the new reference. Free, ~5 sec:

```bash
uv run python scripts/calibrate_from_fixture.py
```

This rewrites `core/scoring/calibration.json`. It pools every
`golden_pred_*.npy` in `tests/fixtures/` and prints what each fixture
would score under the new calibration.

Then restart uvicorn — the calibration is read at import time:

```bash
# Ctrl-C the running uvicorn, then:
uv run uvicorn api.main:app --reload --port 8000
```

Mock mode picks the **first** `golden_pred_*.npy` by sorted (alphabetical)
order. To control which fixture mock returns, name your stems accordingly
or temporarily move others aside.

---

## Common gotchas

**`whisperx` not on PATH** — install with `uv tool install --python 3.11 whisperx`.
Pinning Python 3.11 is required; 3.14 breaks pyannote/torchaudio.

**`uv` warning about `VIRTUAL_ENV` mismatch** — harmless. uv ignores the env
var and uses the project's `.venv`. Silence it by running `deactivate` or
opening a fresh terminal.

**`zsh: command not found: #`** — zsh interactive shells don't treat `#` as
a comment by default. `setopt interactivecomments` once in `~/.zshrc`.

**`gated repo` error** — visit the HF page for the failing repo and click
"Agree and access". Both [facebook/tribev2](https://huggingface.co/facebook/tribev2)
and [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B)
need acceptance.

**Image URL returns 403/410 mid-run** — Instagram and similar CDN URLs
expire. Re-grab the image, or upload to R2 first using `scripts/upload_to_r2.py`.

**Hangs at "Loading TribeModel..."** — first-run download of ~10 GB to
`~/.cache/huggingface/`. Not stuck. Subsequent runs reuse the cache.

**OOM (process killed)** — close everything else (browsers especially).
Drop `--audio-url` if you can; the audio path is the heaviest. For videos,
trim to 30–60 s with `ffmpeg -t 60`.

**`huggingface-cli` deprecated** — use `hf` (the new name).

**Lots of "FutureWarning" / "UserWarning" lines** — harmless, pyannote and
neuralset are noisy. Look at the final shape line and "Wall time" to know
it succeeded:
```
Saved (6, 20484) float32 -> tests/fixtures/golden_pred_post_<hash>_T6.npy
Wall time: 1842s
```

---

## Why fixtures help

A fixture-derived calibration with n=10+ reference clips is the difference
between "Visual Cortex 22" (meaningless against a single sintel baseline)
and "Visual Cortex 75" (meaningful: better than 75% of our library). The
suggestion engine + the score UI both read calibrated values, so growing
the reference pool is the single highest-leverage thing you can do for
score quality before launching publicly.

See [SCALING.md](SCALING.md) for ideas on batching this up on RunPod
($5–10 for 10–15 fixtures vs. ~12 hr unattended on the M3).
