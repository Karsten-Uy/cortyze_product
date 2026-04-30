"""Templated LLM client for free local development.

Returns deterministic suggestions based on which region was flagged and
which content shape the user is analyzing. The shape is detected by
parsing the "Content shape:" line that the prompt builder emits, so
this client doesn't need to be told the type explicitly.

Three template tables — one per content shape (video / post / gallery).
Each region has 2-3 fixes appropriate to that shape. The video templates
are timestamp-friendly; post templates target image / caption / audio;
gallery templates target reorder / swap / drop / add image actions.

# TODO(stage 2): replace with real LLM once prompts are dialed in.
"""

from __future__ import annotations

import re
from typing import Any


# --- Video templates (timestamp-anchored editorial fixes) ----------------

_VIDEO_TEMPLATES: dict[str, list[dict[str, str]]] = {
    "visual_cortex": [
        {
            "title": "Lead with motion in the first second",
            "fix": "Replace the static opening shot with a fast push-in or whip pan. The first second has 2-3x the visual-cortex impact of the rest.",
            "why": "Visual cortex activates strongest on motion onset. Static openings register as background.",
        },
        {
            "title": "Cut twice as often in the dip window",
            "fix": "Shorten clips to 0.8-1.2 seconds during the flagged window. Add a punch-in mid-clip if you can't cut.",
            "why": "Sustained shots over ~2 seconds let visual attention drop off.",
        },
    ],
    "fusiform_face": [
        {
            "title": "Add a face close-up at the dip",
            "fix": "Insert a 2-3 second human reaction shot — eye contact with camera, clearly lit. Even a stock-footage face works.",
            "why": "Fusiform Face Area activates within 200ms of seeing a clear face. Without one, this region barely registers.",
        },
        {
            "title": "Reframe to bring the face larger",
            "fix": "If a face is in frame but small, push in. The face needs to occupy ~25%+ of the frame to drive activation.",
            "why": "Fusiform response scales with face size. Distant faces don't trigger personal-connection signals.",
        },
    ],
    "amygdala": [
        {
            "title": "Add a stakes moment",
            "fix": "Insert a 1-2 second beat showing what happens if the viewer does nothing — frustration, missed opportunity, or surprise. Pair with a sound design hit.",
            "why": "Emotional impact requires felt urgency. Without stakes, the brain treats your content as decorative.",
        },
        {
            "title": "Replace narration with reaction",
            "fix": "Cut the voiceover during the dip and replace with a person reacting — a smile, a flinch, a held breath.",
            "why": "Telling viewers what to feel is less effective than showing someone feeling it.",
        },
    ],
    "prefrontal": [
        {
            "title": "Make the use-case concrete",
            "fix": "Replace abstract benefits with one specific use-case: 'Use this when X happens.' Show the moment, not the category.",
            "why": "Prefrontal evaluation needs concrete scenarios to weigh. Generic benefits don't trigger consideration.",
        },
        {
            "title": "Surface the comparison",
            "fix": "Briefly show what the viewer is doing now vs. what they'd do with your product. Side-by-side or before/after.",
            "why": "The brain can't evaluate without alternatives. Frame the choice explicitly.",
        },
    ],
    "temporal_language": [
        {
            "title": "Slow down the voiceover or add captions",
            "fix": "If voiceover is faster than ~2.5 words/second during the dip, slow to 2.0-2.2 wps OR add burned-in captions with key words highlighted.",
            "why": "Language processing falls behind when speech is too dense. Captions give the brain a second chance.",
        },
        {
            "title": "Cut a redundant beat",
            "fix": "If the dip happens during a list ('available in three colors with eco packaging'), cut to one focused beat instead.",
            "why": "Lists fragment attention. One specific claim lands; three vague ones evaporate.",
        },
    ],
    "hippocampus": [
        {
            "title": "Add a sonic logo or repeated phrase",
            "fix": "Insert a 1-2 second audio signature in the dip — a chord, a tagline repeated, or a specific sound effect tied to your brand.",
            "why": "Memory encoding strengthens with auditory + visual co-activation. A repeated cue at peak moments is what gets remembered.",
        },
        {
            "title": "Reframe with a memorable visual hook",
            "fix": "Bring back a distinctive visual element from earlier in the content — same color, same shape, same person — at the dip moment.",
            "why": "Repetition with variation cues episodic memory. Reusing a visual signature anchors recall.",
        },
    ],
    "motor": [
        {
            "title": "Show the action explicitly",
            "fix": "Insert a 1-second shot of a hand or finger doing the next step — tapping, swiping, reaching. Big and centered.",
            "why": "Motor cortex mirrors observed actions. Without a visible 'action to take,' viewers don't mentally rehearse acting.",
        },
        {
            "title": "End with a single clear CTA",
            "fix": "Strip the closing to one verb + one direction: 'Tap below.' 'Click the link.' Cut anything else.",
            "why": "Decision motor planning requires a singular target. Multiple CTAs split the impulse.",
        },
    ],
    "reward": [
        {
            "title": "Add a payoff moment",
            "fix": "Insert a clear resolution — the satisfied face, the perfect fit, the music drop — right after the dip window.",
            "why": "Reward circuitry needs a release. Build-up without payoff feels unsatisfying and is forgotten.",
        },
        {
            "title": "Match music to the emotional arc",
            "fix": "If the dip happens during static visuals + flat music, swap to a track with a clear lift and time it to a visual change.",
            "why": "Music synchrony with visual beats drives reward activation. Mismatched tracks dampen it.",
        },
    ],
}


# --- Post templates (image / caption / audio levers, no edit timeline) ---

_POST_TEMPLATES: dict[str, list[dict[str, str]]] = {
    "visual_cortex": [
        {
            "title": "Reframe the image with stronger composition",
            "fix": "Re-crop to put the subject on a third, push contrast 10-15%, and cut visual clutter from the edges. The eye should land on one focal point in <300ms.",
            "why": "Visual cortex on a single image is driven by composition and contrast — there's no motion to lean on.",
        },
        {
            "title": "Increase visual hierarchy",
            "fix": "If multiple elements compete for attention, scale the primary subject up 20-30% relative to the rest, or blur the background.",
            "why": "Without motion or cuts, hierarchy is the only way to direct the eye.",
        },
    ],
    "fusiform_face": [
        {
            "title": "Bring a face into the frame",
            "fix": "Re-shoot or re-crop to include a clear, well-lit human face occupying ~25%+ of the image. Eye contact with camera is strongest.",
            "why": "Without a face, this region barely registers — and the post loses the personal-connection signal that drives engagement.",
        },
        {
            "title": "Use a tighter portrait crop",
            "fix": "If a face is in the image but small or partially obscured, re-crop to a head-and-shoulders portrait.",
            "why": "Fusiform response scales with face size and clarity. A wide shot with a tiny face won't trigger it.",
        },
    ],
    "amygdala": [
        {
            "title": "Rewrite the caption hook for urgency",
            "fix": "Replace the opening line with a stakes-loaded prompt — a question, a surprise, or a 'this won't last' framing. First 3 words decide whether the rest gets read.",
            "why": "On a static post the caption is your main lever for emotional impact — the image alone struggles to trigger urgency.",
        },
        {
            "title": "Shoot a more emotionally legible image",
            "fix": "Re-shoot with a clearer emotional cue: a reaction, a moment of surprise, a tactile detail. Calm/neutral images don't fire amygdala signals.",
            "why": "Amygdala needs a felt cue. Decorative product shots register as visual noise.",
        },
    ],
    "prefrontal": [
        {
            "title": "Make the use-case concrete in the caption",
            "fix": "Replace abstract benefits with one specific scenario: 'Use this when X.' Two sentences max. Drop the generic adjectives.",
            "why": "Prefrontal consideration needs concrete situations to weigh. Generic claims don't prompt evaluation.",
        },
        {
            "title": "Show the product in use, not on white",
            "fix": "Swap a studio/white-background image for one showing the product in a real-world context that matches the buyer's life.",
            "why": "Buyers consider products by mentally placing them in their own environment. Studio shots make that harder.",
        },
    ],
    "temporal_language": [
        {
            "title": "Tighten the caption — fewer, sharper words",
            "fix": "Cut the caption by 40-60%. Lead with the most concrete claim. Move hashtags to a comment if they cluster at the end.",
            "why": "Long captions on static posts get skimmed. Tight, specific language lands.",
        },
        {
            "title": "Rewrite for one core idea",
            "fix": "If the caption mixes 2-3 messages (announcement + benefit + CTA), pick the one this post is really about and cut the others.",
            "why": "Language processing fragments across multiple ideas. One idea, said well, beats three said okay.",
        },
    ],
    "hippocampus": [
        {
            "title": "Repeat your brand cue in the caption",
            "fix": "Use your tagline / brand phrase in the first or last line of the caption. Pair it with one distinctive emoji or visual mark you reuse across posts.",
            "why": "Static posts get one shot at memory encoding. Distinctive repetition is what gets recalled days later.",
        },
        {
            "title": "Add a memorable visual signature to the image",
            "fix": "Compose with a recurring color / shape / framing that maps to your brand. The viewer should be able to ID this as yours from a thumbnail.",
            "why": "Recognition from a thumbnail is what drives feed-scroll recall.",
        },
    ],
    "motor": [
        {
            "title": "Add an explicit, single CTA in the caption",
            "fix": "End with one verb + one direction: 'Comment your favorite below.' 'Tap the link in bio.' One CTA only.",
            "why": "Static posts have no visual action cue, so the caption's verb is the only motor trigger. Multiple CTAs split the impulse.",
        },
        {
            "title": "Show a hand or finger in the image",
            "fix": "Re-shoot with a hand interacting with the product — pointing, holding, tapping. Don't hide the action.",
            "why": "Motor cortex mirrors observed actions. A hand in frame primes the viewer to imagine acting themselves.",
        },
    ],
    "reward": [
        {
            "title": "Add audio that lands the payoff",
            "fix": "If you can attach an audio track, pick one with a clear lift in the second half. Static posts with no audio leave the reward circuit unfed.",
            "why": "Reward needs release. Music synchrony is a fast way to deliver it on a single image.",
        },
        {
            "title": "Reshoot for a 'satisfaction' moment",
            "fix": "Capture the post-purchase / mid-use / before-after frame instead of the product alone — the result, not the object.",
            "why": "Reward fires on resolution. The object alone is anticipation; the satisfied moment is the payoff.",
        },
    ],
}


# --- Gallery templates (reorder / swap / drop / add image levers) --------

_GALLERY_TEMPLATES: dict[str, list[dict[str, str]]] = {
    "visual_cortex": [
        {
            "title": "Lead with your highest-contrast image",
            "fix": "Reorder so the image with the strongest visual hierarchy (most contrast, biggest subject, cleanest composition) sits at position 1.",
            "why": "First-image attention sets the ceiling for the whole gallery. A weak lead drags every downstream score.",
        },
        {
            "title": "Drop the lowest-contrast image",
            "fix": "Look at the per-image bars — the image at the bottom is pulling the visual cortex average down. Remove it; 4 strong images beat 5 mixed ones.",
            "why": "Galleries average per-image attention. One filler image hurts every region's score.",
        },
    ],
    "fusiform_face": [
        {
            "title": "Add a face close-up",
            "fix": "Insert a tight head-and-shoulders portrait (eye contact, clearly lit) as the second or third image. Even one face in the carousel lifts this region substantially.",
            "why": "Fusiform fires within 200ms on a clear face. A face-less gallery leaves this signal flat across all images.",
        },
        {
            "title": "Swap the weakest face image",
            "fix": "If a gallery image has a face that's small, blurred, or off-center, replace it with a closer, sharper alternate. The face should occupy 25%+ of frame.",
            "why": "Distant faces don't trigger the personal-connection response. Replacing one weak face image lifts the regional average.",
        },
    ],
    "amygdala": [
        {
            "title": "Lead with the most emotionally legible image",
            "fix": "Reorder so the image with the clearest emotion (a reaction, a stakes moment, a surprise) sits at position 1.",
            "why": "Lead-image emotion sets the gallery's emotional baseline. Calm openings make every later image feel decorative.",
        },
        {
            "title": "Add a stakes image mid-gallery",
            "fix": "Insert a 'what if you don't' image — frustration, missed moment, before-shot — between your benefit and your CTA images.",
            "why": "Pure benefit galleries register as flat. Contrast against a negative beat makes the resolution feel earned.",
        },
    ],
    "prefrontal": [
        {
            "title": "Add a use-case image",
            "fix": "Insert an image showing the product in a specific real-world scenario the buyer would recognize. One concrete scene, not a generic showcase.",
            "why": "Prefrontal consideration needs concrete situations. Galleries that only show the object don't trigger 'how would I use this?'",
        },
        {
            "title": "Reorder to lead with a comparison",
            "fix": "Put a before/after or with/without image first, then move to single-product shots.",
            "why": "Comparison frames a choice. Galleries without one don't activate evaluation.",
        },
    ],
    "temporal_language": [
        {
            "title": "Tighten the caption to one core idea",
            "fix": "Galleries are visual-first. Cut the caption to a single sentence that anchors the throughline of the images. Move hashtags to a reply.",
            "why": "Long captions on visual carousels get skimmed; the visual narrative carries most of the load.",
        },
        {
            "title": "Add text-overlay hooks on key images",
            "fix": "Burn 1-3 word labels onto images 1, 3, and the final image — short enough to read in <500ms while swiping.",
            "why": "Reading captures language activation per-image. Short on-image text outperforms a long external caption.",
        },
    ],
    "hippocampus": [
        {
            "title": "Close with your brand signature",
            "fix": "End the gallery on an image that uses your distinctive color / shape / mark. The closing image is what the viewer remembers.",
            "why": "Recall is closing-image-weighted in carousels. A generic final image wastes the strongest memory slot.",
        },
        {
            "title": "Repeat a visual motif across images",
            "fix": "Pick one element (a color, a prop, a framing rule) and use it in 3+ images. Repetition with variation is what cues episodic memory.",
            "why": "Galleries have built-in repetition slots — using them is free memory boost.",
        },
    ],
    "motor": [
        {
            "title": "Make the final image the CTA",
            "fix": "Reorder so the last image is the explicit action shot — a finger tapping, a hand reaching, or text saying the next step. One verb, one direction.",
            "why": "Carousel CTAs land on the closing image. If the last image is decorative, the swipe momentum dies without action.",
        },
        {
            "title": "Drop images that don't move the story forward",
            "fix": "Cut any image that doesn't either (a) hook attention, (b) build the case, or (c) drive action. 4 purposeful images > 7 mixed.",
            "why": "Each unnecessary image is one more swipe before the CTA. Motor planning needs forward momentum.",
        },
    ],
    "reward": [
        {
            "title": "End on a payoff image",
            "fix": "Make the final image the satisfaction moment — the result, the after, the smile. Anything earlier than the close gets buried.",
            "why": "Reward circuitry fires on resolution. Carousels that close mid-pitch leave the brain hanging.",
        },
        {
            "title": "Add audio with a lift in the second half",
            "fix": "Attach a track that builds — quiet first half, payoff in the second. Pair the music's lift with your strongest visual.",
            "why": "Audio synchrony with the gallery's narrative arc multiplies reward activation.",
        },
    ],
}


# The prompt builder emits one of these shape tags in "Content shape:";
# multi-image posts use "carousel" so the mock can reach the right table.
_TEMPLATES_BY_SHAPE: dict[str, dict[str, list[dict[str, str]]]] = {
    "video": _VIDEO_TEMPLATES,
    "post": _POST_TEMPLATES,
    "carousel": _GALLERY_TEMPLATES,
}


class MockLLMClient:
    """Returns templated suggestions appropriate to the content shape.

    Detects shape from the "Content shape: <tag>" line emitted by the
    prompt builder. Adds an image-range anchor for carousels (multi-image
    posts) or a timestamp window for video / audio-bearing single-image
    posts.

    Deterministic, free, always succeeds — exactly what frontend dev and
    tests need.
    """

    def chat_json(self, *, system: str, user: str) -> Any:
        shape = _extract_content_shape(user)
        templates_table = _TEMPLATES_BY_SHAPE.get(shape, _VIDEO_TEMPLATES)
        region = _extract_region(user)
        templates = templates_table.get(region, [])[:3]
        if not templates:
            return [
                {
                    "title": "Region needs attention",
                    "fix": f"Specific advice for region {region!r} requires the live LLM.",
                    "why": "Mock client only carries templates for the 8 marketing regions.",
                }
            ]

        out: list[dict[str, Any]] = []
        if shape == "carousel":
            start_idx, end_idx = _extract_image_range(user)
            for t in templates:
                row: dict[str, Any] = dict(t)
                if start_idx is not None:
                    row["image_index_start"] = start_idx
                    row["image_index_end"] = end_idx if end_idx is not None else start_idx
                out.append(row)
        else:
            ts_start, ts_end = _extract_timestamp(user)
            for t in templates:
                row = dict(t)
                if ts_start is not None and ts_end is not None:
                    row["timestamp_start_s"] = ts_start
                    row["timestamp_end_s"] = ts_end
                out.append(row)
        return out


def _extract_content_shape(user_prompt: str) -> str:
    match = re.search(
        r"\bcontent\s*shape[:\s]+([a-z_]+)", user_prompt, re.IGNORECASE
    )
    return match.group(1).lower() if match else "video"


def _extract_region(user_prompt: str) -> str:
    match = re.search(r"\bregion[:\s]+([a-z_]+)", user_prompt, re.IGNORECASE)
    return match.group(1).lower() if match else ""


def _extract_timestamp(user_prompt: str) -> tuple[float | None, float | None]:
    """Pull the first 'M:SS-M:SS' or '<int>-<int>s' window from the prompt."""
    m = re.search(
        r"(\d+):(\d{2})\s*[-–]\s*(\d+):(\d{2})", user_prompt
    )
    if m:
        a = int(m.group(1)) * 60 + int(m.group(2))
        b = int(m.group(3)) * 60 + int(m.group(4))
        return float(a), float(b)
    # Matches both "14-19s" and "14s-19s" so the prompt format can vary.
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*s?\s*[-–]\s*(\d+(?:\.\d+)?)\s*s\b", user_prompt
    )
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


def _extract_image_range(user_prompt: str) -> tuple[int | None, int | None]:
    """Pull the first 'image N' or 'images N-M' reference from the prompt."""
    m = re.search(
        r"\bimages?\s+(\d+)\s*[-–]\s*(\d+)", user_prompt, re.IGNORECASE
    )
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"\bimages?\s+(\d+)", user_prompt, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(1))
    return None, None
