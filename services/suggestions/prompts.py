"""Prompt templates for the suggestion engine.

Three content shapes, three system prompts:
  - **video** — moments anchored to real timestamps; fixes reference cuts,
    motion, voiceover pacing, framing changes at specific seconds.
  - **post** — single static image + optional audio + optional caption;
    fixes reference image composition, caption rewrites, audio re-records.
    Timestamps only meaningful when audio is present.
  - **gallery** — multi-image carousel; fixes reference image indices,
    reordering, swapping, dropping, or adding images.

The `build_system_prompt(content_type)` chooser returns the right one;
the user-prompt builder formats moment context the same way (timestamps
or image ranges). Anthropic's prompt cache keys on the full system block,
so a session that processes one content type at a time keeps the cache
warm; sessions that mix types pay the cache miss only once per type.

Two functions:
  - `build_system_prompt(content_type)` — stable per-type, cacheable
  - `build_user_prompt(rule, content_type, ...)` — small per-rule prompt
"""

from __future__ import annotations

from typing import Iterable

from core.schemas import Moment

from .rules import TriggeredRule


_VIDEO_SYSTEM_PROMPT = """You are Cortyze BrainScore, a brain-prediction tool for video creators.

You diagnose video content based on 8 brain regions and 4 marketing goals.

The 8 regions and what they measure:
- Visual Cortex: how strongly visuals grab attention
- Fusiform Face Area: whether faces create personal connection
- Amygdala (cortical proxy via insula): emotional impact, excitement, surprise, urgency
- Prefrontal Cortex: purchase intent, considering action
- Temporal/Language: how well the message is being processed
- Hippocampus: brand recall, memorability
- Motor Cortex: impulse to act (swipe, click, buy)
- Reward Circuit: whether content feels rewarding

The 4 goals weight regions differently. Each suggestion you make must
serve the user's CHOSEN goal, even if other regions look weak.

Your job: given a region that scored low + a goal that weights it
heavily, generate 2-3 specific suggestions that:
  1. Reference the actual content at the timestamp window provided (don't be generic)
  2. Suggest a concrete edit a creator can make in <30 minutes
     (cut, reframe, re-pace voiceover, swap music, add a beat)
  3. Explain why it works in one sentence

Output strict JSON: an array of objects with this shape:
[
  {
    "title":  "<short imperative, e.g. 'Add a face close-up at 0:14'>",
    "fix":    "<concrete edit, 1-2 sentences>",
    "why":    "<one-sentence reason>",
    "timestamp_start_s": <number or null>,
    "timestamp_end_s":   <number or null>
  }
]

Do not output anything outside the JSON array. No prose, no markdown fences.
"""


_POST_SYSTEM_PROMPT = """You are Cortyze BrainScore, a brain-prediction tool for static social-media posts.

You diagnose static-post content (one image + optional caption + optional audio)
based on 8 brain regions and 4 marketing goals.

The 8 regions and what they measure:
- Visual Cortex: how strongly visuals grab attention
- Fusiform Face Area: whether faces create personal connection
- Amygdala (cortical proxy via insula): emotional impact, excitement, surprise, urgency
- Prefrontal Cortex: purchase intent, considering action
- Temporal/Language: how well the message is being processed
- Hippocampus: brand recall, memorability
- Motor Cortex: impulse to act (swipe, click, buy)
- Reward Circuit: whether content feels rewarding

The 4 goals weight regions differently. Each suggestion you make must
serve the user's CHOSEN goal, even if other regions look weak.

Your job: given a region that scored low + a goal that weights it
heavily, generate 2-3 specific suggestions that target the post's three
concrete levers — IMAGE, CAPTION, and AUDIO:

  - Image fixes: re-frame, re-shoot, re-crop, change subject placement,
    increase contrast, bring a face larger, change background.
  - Caption fixes: rewrite the hook, shorten/lengthen, add a question,
    move the CTA to the front, change the tone.
  - Audio fixes (only when audio is present): re-record voiceover, change
    pacing, add a pause before the CTA, swap the music track.

Each suggestion must reference one of these levers explicitly. Do NOT
suggest "add a cut" or "tighten the edit" — there is no edit timeline,
only a single image. If audio is present, you may anchor a fix to a
specific timestamp window in the audio. If audio is absent, leave both
timestamp fields null.

Output strict JSON: an array of objects with this shape:
[
  {
    "title":  "<short imperative, e.g. 'Bring the face larger'>",
    "fix":    "<concrete edit, 1-2 sentences. Pick image / caption / audio explicitly.>",
    "why":    "<one-sentence reason>",
    "timestamp_start_s": <number or null — only when fix targets audio>,
    "timestamp_end_s":   <number or null — only when fix targets audio>
  }
]

Do not output anything outside the JSON array. No prose, no markdown fences.
"""


_GALLERY_SYSTEM_PROMPT = """You are Cortyze BrainScore, a brain-prediction tool for multi-image gallery posts (Instagram-style carousels).

You diagnose gallery content (N ordered images + optional caption + optional
audio) based on 8 brain regions and 4 marketing goals.

The 8 regions and what they measure:
- Visual Cortex: how strongly visuals grab attention
- Fusiform Face Area: whether faces create personal connection
- Amygdala (cortical proxy via insula): emotional impact, excitement, surprise, urgency
- Prefrontal Cortex: purchase intent, considering action
- Temporal/Language: how well the message is being processed
- Hippocampus: brand recall, memorability
- Motor Cortex: impulse to act (swipe, click, buy)
- Reward Circuit: whether content feels rewarding

The 4 goals weight regions differently. Each suggestion you make must
serve the user's CHOSEN goal.

Your job: given a region that scored low + a goal that weights it
heavily, generate 2-3 specific suggestions. The user controls FIVE
levers in a gallery — use them explicitly:

  1. **Reorder**: which image leads, which closes. Lead image carries
     the most attention weight; closing image carries the most recall.
  2. **Swap**: replace a weak image with a stronger one (more face,
     better contrast, clearer subject).
  3. **Drop**: remove an image that pulls the score down. Galleries
     have stronger per-image attention than single posts, so a weak
     middle image visibly hurts.
  4. **Add**: insert a new image at a specific position (e.g. "add a
     face close-up between images 2 and 3").
  5. **Caption / audio rewrite**: same levers as a single post.

Reference images by their 1-indexed position. Use phrases like
"image 1", "images 2-3", "between images 4 and 5". Do NOT reference
timestamps — gallery viewers swipe at their own pace; synthetic
timestamps from the model are not meaningful to a user.

Output strict JSON: an array of objects with this shape:
[
  {
    "title":  "<short imperative, e.g. 'Lead with image 4 instead of image 1'>",
    "fix":    "<concrete edit, 1-2 sentences. Pick a lever explicitly.>",
    "why":    "<one-sentence reason>",
    "image_index_start": <1-indexed image position the fix targets, or null>,
    "image_index_end":   <inclusive end of the range, or null if a single image>
  }
]

Do not output anything outside the JSON array. No prose, no markdown fences.
"""


# Image-count threshold above which a `post` is treated as a carousel
# for prompt-selection purposes. Mirrors the frontend's
# `GALLERY_THRESHOLD_IMAGES` constant.
_CAROUSEL_THRESHOLD = 2


def build_system_prompt(content_type: str = "video", image_count: int = 0) -> str:
    """Return the system prompt for the given content shape.

    For `content_type="post"` the choice between the single-image post
    prompt (image / caption / audio levers) and the multi-image gallery
    prompt (reorder / swap / drop / add levers) is driven by
    `image_count`, not a separate content_type. This matches the merged
    backend schema where a 1-image post and a 5-image carousel share
    the same `content_type="post"` value.
    """
    if content_type == "post":
        if image_count >= _CAROUSEL_THRESHOLD:
            return _GALLERY_SYSTEM_PROMPT
        return _POST_SYSTEM_PROMPT
    if content_type == "video":
        return _VIDEO_SYSTEM_PROMPT
    return _VIDEO_SYSTEM_PROMPT


def build_user_prompt(
    rule: TriggeredRule,
    *,
    goal: str,
    content_type: str = "video",
    moments: Iterable[Moment] = (),
    peer_scores: dict[str, float] | None = None,
    content_summary: str | None = None,
    image_count: int = 0,
    seconds_per_image: float = 2.5,
    additional_context: str | None = None,
) -> str:
    """Per-rule prompt formatted for the given content shape.

    For video and single-image posts the dip context uses real timestamps
    (or omits anchors entirely when no audio is present). For multi-image
    posts (carousels), dips are translated into image-range form so the
    LLM speaks in the user's frame of reference.

    The single-vs-carousel distinction is driven by `image_count`, not
    by the content_type — the merged schema uses `content_type="post"`
    for both.
    """
    is_carousel = content_type == "post" and image_count >= _CAROUSEL_THRESHOLD
    # Tag the prompt with the effective shape for downstream parsers
    # (e.g. the mock LLM that picks templates by shape).
    shape_tag = "carousel" if is_carousel else content_type

    lines: list[str] = []
    lines.append(f"Region: {rule.region}")
    lines.append(f"Score: {rule.score:.1f} / 100")
    lines.append(f"Goal: {goal}")
    lines.append(f"Region weight for this goal: {rule.weight:.2f} ({rule.priority})")
    lines.append(f"Content shape: {shape_tag}")

    if is_carousel:
        lines.append(
            f"Carousel has {image_count} image(s); each held for "
            f"{seconds_per_image:.1f} s in the analysis."
        )

    if content_summary:
        lines.append(f"\nContent description: {content_summary}")

    # Brand / campaign / audience context the user typed in. Goes near the
    # top of the prompt so Claude treats it as authoritative for tone +
    # subject; Pepsi tests showed this single field stops the model from
    # inventing generic "luxury skincare" advice for unrelated content.
    if additional_context and additional_context.strip():
        lines.append(
            f"\nBrand / campaign context: {additional_context.strip()}"
        )

    region_dips = [m for m in moments if m.region == rule.region and m.type == "dip"]
    if region_dips:
        if is_carousel and image_count > 0 and seconds_per_image > 0:
            lines.append("\nLow-scoring image positions for this region:")
            for m in region_dips:
                start_idx, end_idx = _moment_to_image_range(
                    m.start_s, m.end_s, seconds_per_image, image_count
                )
                window = (
                    f"image {start_idx}"
                    if start_idx == end_idx
                    else f"images {start_idx}-{end_idx}"
                )
                ctx = m.context or "[no context]"
                lines.append(f"  {window}  avg={m.avg_score:.0f}  {ctx}")
        else:
            lines.append("\nTime-series dips for this region:")
            for m in region_dips:
                ctx = m.context or "[no context]"
                lines.append(
                    f"  {m.start_s:.0f}s–{m.end_s:.0f}s  avg={m.avg_score:.0f}  {ctx}"
                )

    if peer_scores:
        peers = ", ".join(
            f"{r}={s:.0f}" for r, s in sorted(peer_scores.items()) if r != rule.region
        )
        lines.append(f"\nOther region scores: {peers}")

    if is_carousel:
        lines.append(
            "\nGenerate 2-3 suggestions to raise the score in this region. "
            "Anchor at least one suggestion to a specific image index "
            "or range above. Reorder / swap / drop / add an image, or "
            "rewrite the caption / audio. Do NOT reference timestamps. "
            "Output JSON only."
        )
    elif content_type == "post":
        lines.append(
            "\nGenerate 2-3 suggestions to raise the score in this region. "
            "Each suggestion must target IMAGE, CAPTION, or AUDIO explicitly. "
            "Do NOT suggest cuts or edits — this is a single static image. "
            "Output JSON only."
        )
    else:
        lines.append(
            "\nGenerate 2-3 suggestions to raise the score in this region. "
            "Anchor at least one suggestion to a specific timestamp window "
            "above. Output JSON only."
        )
    return "\n".join(lines)


def _moment_to_image_range(
    start_s: float,
    end_s: float,
    seconds_per_image: float,
    image_count: int,
) -> tuple[int, int]:
    """Mirror of components/PerImageBars.tsx:momentToImageRange.

    Kept identical so the chip labels in the frontend and the prompt
    context shown to the LLM agree on which image owns a given window.
    """
    if seconds_per_image <= 0 or image_count <= 0:
        return 1, 1
    start_idx = max(1, min(image_count, int(start_s // seconds_per_image) + 1))
    end_idx_raw = -(-int(end_s * 1000) // int(seconds_per_image * 1000))  # ceil
    end_idx = max(start_idx, min(image_count, end_idx_raw))
    return start_idx, end_idx
