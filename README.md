# Player-ID Remapper

Re-identifies every player across two asynchronous camera feeds  
(TV “broadcast” + high-angle “tacticam”) and writes three annotated
videos:

| File | Purpose |
|------|---------|
| `output_broadcast.mp4` | Broadcast view with **StrongSORT IDs** |
| `output_tacticam.mp4`  | Tacticam view with native IDs |
| `remapped_tacticam.mp4`| Tacticam view **remapped ↔ broadcast IDs** |

<p align="center">
  <img src="results/ss1.png" width="600">
</p>

---

## Pipeline

1. **Detection** – YOLO-v11 (`best.pt`) @ every *N* frames  
2. **Multi-object tracking** – StrongSORT (`n_init=1`, `max_age=30`)  
3. **Appearance embedding** – OSNet-x0.25 (`osnet_x0_25_msmt17.pt`)  
4. **Feature bank** – median-pooled 1024-D vectors *(frame-stamped)*  
5. **Matching** – Hungarian solver with  
   *temporal gate* ± `WIN_SEC` seconds & cosine cutoff `THR`  
6. **TTL Overlay** – last box kept `TTL_FRAMES` to hide tracker flicker  

See [`docs/pipeline.png`](docs/pipeline.png) for the full diagram.

---

## Quick start

```bash
git clone https://github.com/yourname/player-remap.git
cd player-remap
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# place your two input clips in examples/ or pass absolute paths
python tracker.py \
    --broadcast examples/broadcast.mp4 \
    --tacticam  examples/tacticam.mp4
