"""Generate the (20484,) Desikan-Killiany label array on fsaverage5.

TRIBE v2 outputs predictions on the fsaverage5 surface mesh (10242 vertices
per hemisphere, 20484 total). To aggregate per-vertex predictions into the
8 marketing-named brain regions in core.atlas.regions, we need a (20484,)
integer array assigning each vertex to a DK atlas label.

nilearn ships the Destrieux 2010 atlas on fsaverage5 directly; Destrieux is
finer-grained (~75 labels/hemi) than DK (~35), so we apply a hand-curated
Destrieux -> DK mapping to project Destrieux labels back to the DK names
used in core/atlas/regions.py. Approximate but sufficient for Stage 1
placeholder-quality scoring.

# TODO(stage 2): replace with proper FreeSurfer aparc.annot projection if
# higher anatomical fidelity is needed for ad-grade brain region scoring.

Outputs to core/atlas/data/:
- fsaverage5_dk_labels.npy   (20484,) int32, each entry is a DK label ID
- fsaverage5_dk_labels.json  sidecar: dk_label_name -> id, mapping notes
"""

import json
import sys
from pathlib import Path

import numpy as np
from nilearn import datasets

# Hand-curated Destrieux 2010 -> Desikan-Killiany approximate mapping.
# Sources: Destrieux et al. 2010, FreeSurfer docs, Klein & Tourville 2012.
# Many-to-one is expected (Destrieux subdivides DK regions further).
# Destrieux labels not appearing here become "unknown".
DESTRIEUX_TO_DK: dict[str, str] = {
    # Visual cortex
    "S_calcarine": "pericalcarine",
    "G_cuneus": "cuneus",
    "G_occipital_middle": "lateraloccipital",
    "G_occipital_sup": "lateraloccipital",
    "S_oc_middle_and_Lunatus": "lateraloccipital",
    "S_oc_sup_and_transversal": "lateraloccipital",
    "Pole_occipital": "lateraloccipital",
    "G_and_S_occipital_inf": "lateraloccipital",
    "S_occipital_ant": "lateraloccipital",
    "G_oc-temp_med-Lingual": "lingual",
    "S_oc-temp_med_and_Lingual": "lingual",
    "S_collat_transv_post": "lingual",
    # Fusiform face area
    "G_oc-temp_lat-fusifor": "fusiform",
    "S_oc-temp_lat": "fusiform",
    # Insula (amygdala proxy per core/atlas/regions.py)
    "G_insular_short": "insula",
    "G_Ins_lg_and_S_cent_ins": "insula",
    "S_circular_insula_ant": "insula",
    "S_circular_insula_inf": "insula",
    "S_circular_insula_sup": "insula",
    "Lat_Fis-ant-Horizont": "insula",
    "Lat_Fis-ant-Vertical": "insula",
    # Prefrontal
    "G_front_sup": "superiorfrontal",
    "S_front_sup": "superiorfrontal",
    "G_front_middle": "rostralmiddlefrontal",
    "S_front_middle": "rostralmiddlefrontal",
    "G_front_inf-Opercular": "parsopercularis",
    "S_front_inf": "parsopercularis",
    "G_front_inf-Triangul": "parstriangularis",
    "G_front_inf-Orbital": "parsorbitalis",
    "G_orbital": "lateralorbitofrontal",
    "S_orbital_lateral": "lateralorbitofrontal",
    "S_orbital-H_Shaped": "lateralorbitofrontal",
    "G_rectus": "medialorbitofrontal",
    "S_orbital_med-olfact": "medialorbitofrontal",
    "S_suborbital": "medialorbitofrontal",
    "G_subcallosal": "medialorbitofrontal",
    "G_and_S_transv_frontopol": "frontalpole",
    "G_and_S_frontomargin": "frontalpole",
    # Temporal / language
    "G_temp_sup-Lateral": "superiortemporal",
    "G_temp_sup-Plan_polar": "superiortemporal",
    "G_temp_sup-Plan_tempo": "superiortemporal",
    "S_temporal_sup": "bankssts",
    "G_temporal_middle": "middletemporal",
    "S_temporal_inf": "middletemporal",
    "G_temporal_inf": "inferiortemporal",
    "Pole_temporal": "inferiortemporal",
    "G_temp_sup-G_T_transv": "transversetemporal",
    "S_temporal_transverse": "transversetemporal",
    # Hippocampus group (parahippocampal / entorhinal)
    "G_oc-temp_med-Parahip": "parahippocampal",
    "S_collat_transv_ant": "entorhinal",
    # Motor / action
    "G_precentral": "precentral",
    "S_precentral-inf-part": "precentral",
    "S_precentral-sup-part": "precentral",
    "G_postcentral": "postcentral",
    "S_postcentral": "postcentral",
    "S_central": "postcentral",
    "G_and_S_paracentral": "paracentral",
    "G_and_S_subcentral": "postcentral",
    # Reward circuit (cingulate)
    "G_and_S_cingul-Ant": "rostralanteriorcingulate",
    "S_pericallosal": "rostralanteriorcingulate",
    "G_and_S_cingul-Mid-Ant": "caudalanteriorcingulate",
    "G_and_S_cingul-Mid-Post": "posteriorcingulate",
    "S_cingul-Marginalis": "posteriorcingulate",
    "S_subparietal": "posteriorcingulate",
    "G_cingul-Post-dorsal": "isthmuscingulate",
    "G_cingul-Post-ventral": "isthmuscingulate",
}


def main() -> int:
    here = Path(__file__).resolve().parent
    output_dir = here.parent / "core" / "atlas" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching Destrieux atlas on fsaverage5 (first run downloads ~few MB)...", flush=True)
    dest = datasets.fetch_atlas_surf_destrieux()

    raw_labels = list(dest["labels"])
    dest_label_names = [l.decode() if isinstance(l, bytes) else str(l) for l in raw_labels]

    map_left = np.asarray(dest["map_left"])
    map_right = np.asarray(dest["map_right"])
    if map_left.shape != (10242,) or map_right.shape != (10242,):
        print(
            f"ERROR: unexpected hemisphere shapes lh={map_left.shape} rh={map_right.shape}",
            file=sys.stderr,
        )
        return 1

    # Convention used by tribev2.plotting: [lh_0..lh_10241, rh_0..rh_10241]
    dest_per_vertex = np.concatenate([map_left, map_right]).astype(np.int32)
    assert dest_per_vertex.shape == (20484,)

    dk_labels_used = sorted(set(DESTRIEUX_TO_DK.values()) | {"unknown"})
    dk_label_to_id = {name: i for i, name in enumerate(dk_labels_used)}
    unknown_id = dk_label_to_id["unknown"]

    # Build Destrieux ID -> DK ID lookup, then vectorized index.
    dest_to_dk = np.full(len(dest_label_names), unknown_id, dtype=np.int32)
    for did, dest_name in enumerate(dest_label_names):
        dk_name = DESTRIEUX_TO_DK.get(dest_name)
        if dk_name is not None:
            dest_to_dk[did] = dk_label_to_id[dk_name]

    dk_per_vertex = dest_to_dk[dest_per_vertex]
    n_mapped = int((dk_per_vertex != unknown_id).sum())
    pct = 100.0 * n_mapped / 20484

    npy_path = output_dir / "fsaverage5_dk_labels.npy"
    json_path = output_dir / "fsaverage5_dk_labels.json"
    np.save(npy_path, dk_per_vertex)
    json_path.write_text(
        json.dumps(
            {
                "shape": list(dk_per_vertex.shape),
                "dtype": str(dk_per_vertex.dtype),
                "vertex_order": "lh_0..lh_10241, rh_0..rh_10241 (concatenated, fsaverage5)",
                "source_atlas": "nilearn.datasets.fetch_atlas_surf_destrieux",
                "mapping_note": "Hand-curated Destrieux 2010 -> Desikan-Killiany approximate mapping; many-to-one. See scripts/build_atlas_labels.py for the table.",
                "label_to_id": dk_label_to_id,
                "vertices_mapped_pct": pct,
            },
            indent=2,
        )
        + "\n"
    )

    print(f"Saved {npy_path}: ({dk_per_vertex.shape[0]},) {dk_per_vertex.dtype}")
    print(f"Saved {json_path}: {len(dk_labels_used)} DK labels (incl. 'unknown')")
    print(
        f"Mapped {n_mapped}/{20484} vertices ({pct:.1f}%) "
        "- rest are 'unknown' (medial wall, parietal regions outside the 8 marketing regions, etc.)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
