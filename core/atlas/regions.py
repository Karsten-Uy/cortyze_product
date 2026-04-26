"""8-region grouping for marketing-relevant cortical activity.

Single source of truth for which Desikan-Killiany atlas labels belong to
which marketing-named region. Consumed by core.atlas.mapper and
core.scoring.goals; the same keys appear in BrainReport.region_scores and
must stay in sync across all three.
"""

REGIONS: dict[str, list[str]] = {
    "visual_cortex": [
        "pericalcarine",
        "cuneus",
        "lateraloccipital",
        "lingual",
    ],
    "fusiform_face": [
        "fusiform",
    ],
    # TRIBE v2 has no cortical-surface vertex for the amygdala (subcortical).
    # Insula is a documented cortical proxy per IMPLEMENTATION_PLAN.md §8;
    # revisit with ground-truth ad data in Stage 2.
    "amygdala": [
        "insula",
    ],
    "prefrontal": [
        "superiorfrontal",
        "rostralmiddlefrontal",
        "caudalmiddlefrontal",
        "parsopercularis",
        "parstriangularis",
        "parsorbitalis",
        "lateralorbitofrontal",
        "medialorbitofrontal",
        "frontalpole",
    ],
    "temporal_language": [
        "superiortemporal",
        "middletemporal",
        "inferiortemporal",
        "bankssts",
        "transversetemporal",
    ],
    "hippocampus": [
        "parahippocampal",
        "entorhinal",
    ],
    "motor": [
        "precentral",
        "postcentral",
        "paracentral",
    ],
    "reward": [
        "rostralanteriorcingulate",
        "caudalanteriorcingulate",
        "isthmuscingulate",
        "posteriorcingulate",
    ],
}
