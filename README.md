# Player-ID Remapper

Re-identifies every player across two asynchronous camera feeds  
(TV “broadcast” + high-angle “tacticam”) and writes three annotated
videos:

| File | Purpose |
|------|---------|
| `output_broadcast.mp4` | Broadcast view with **StrongSORT IDs** |
| `output_tacticam.mp4`  | Tacticam view with native IDs |
| `remapped_tacticam.mp4`| Tacticam view **remapped ↔ broadcast IDs** |

<p align="left">
  <img src="results/ss1.png" width="600">
  <figcaption>Broadcast.mp4 ID's ☝️</figcaption>
</p>
<p align="left">
  <img src="results/ss2.png" width="600">
  <figcaption>Remapped Tacticam.mp4 ID's to Broadcast ID ☝️</figcaption>
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


---

## Quick start

```bash
#Clone the git
git clone https://github.com/Varun1319/Cross-Camera-Player-Mapping

#Change the directory
cd Cross-Camera-Player-Mapping

#Create a virtual environment
python3 -m venv .venv && source .venv/bin/activate

#Install requirements
pip install -r requirements.txt

# Place the sample videos,YOLOv11 model in the file and give the same path and run
python tracker.py
```



