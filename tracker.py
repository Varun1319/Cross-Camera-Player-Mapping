# StrongSORT + YOLOv11 two-camera ID-remapping
# Note: The script assumes the YOLOv11 model and StrongSORT weights are available in the specified paths.
# Make sure to adjust the paths and parameters according to your setup.
# The script also assumes the input videos are named "broadcast.mp4" and "tacticam.mp4".
# The output videos will be saved as "output_broadcast.mp4", "output_tacticam.mp4", and "remapped_tacticam.mp4".
# The tracks from the tacticam will be saved in "tacticam_tracks.pkl". 

from ultralytics import YOLO
from strongsort.strong_sort import StrongSORT
import cv2, torch, numpy as np, pickle, os
from pathlib import Path
from scipy.spatial.distance import cosine
import torchreid
from torchvision import transforms

# Labelling 
LABEL_MAP = {0: 'Player', 1: 'GK', 2: 'P', 3: 'R'}
COLOR_MAP = {0: (0,255,255), 
             1: (0,0,255),    
             2: (0,255,0),     
             3: (255,0,0)}     

# RE-ID backbone (torchreid)
REID_W   = Path("strongsort/weights/osnet_x0_25_msmt17.pt")
device   = "cuda" if torch.cuda.is_available() else "cpu"
NUM_ID_MSMT17 = 1041

reid = torchreid.models.build_model("osnet_x0_25", num_classes=NUM_ID_MSMT17,pretrained=False)
torchreid.utils.load_pretrained_weights(reid, str(REID_W))
reid.eval().to(device)

prep = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize((256,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def feat_of(crop):
    if crop.size == 0: return None
    with torch.no_grad():
        return reid(prep(crop).unsqueeze(0).to(device)).cpu().numpy().flatten()

# Detector & trackers
detector   = YOLO("best.pt")
tracker_b  = StrongSORT(REID_W, device=device, fp16=False,n_init=1, max_age=90)
tracker_t  = StrongSORT(REID_W, device=device, fp16=False,n_init=1, max_age=90)

# time-to-live for boxes
TTL_FRAMES = 15          

def run_video(src, tracker, dst, save_tracks=False, skip=2, conf=0.25):
    cap = cv2.VideoCapture(src)
    w,h,fps = int(cap.get(3)), int(cap.get(4)), cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))

    bank, frames, fid = {}, {}, 0
    last_box = {}                       
    while True:
        ok, fr = cap.read()
        if not ok: break
        step = []
        # detect every `skip`-th frame
        if fid % skip == 0:
            res  = detector.predict(fr, conf=conf, iou=0.4, verbose=False)[0]
            dets, balls = [], []
            BALL_CONF, BALL_MAX = 0.50, 70

            for x1,y1,x2,y2,conf_sc,cls in res.boxes.data.cpu().numpy():
                cls = int(cls)
                if cls not in LABEL_MAP: continue
                dets.append([x1,y1,x2,y2,conf_sc,cls])

            dets_tensor = (torch.as_tensor(dets, dtype=torch.float32)
                           if dets else torch.empty((0,6), dtype=torch.float32))
            rows = tracker.update(dets_tensor, fr)

            # draw tracked people
            for x1,y1,x2,y2,tid,score,cls in rows:
                x1,y1,x2,y2 = map(int,(x1,y1,x2,y2))
                tid, cls    = int(tid), int(cls)
                col, lbl    = COLOR_MAP.get(cls,(255,255,255)), LABEL_MAP[cls]

                feat = feat_of(fr[y1:y2, x1:x2])
                if feat is not None:
                    bank.setdefault(tid, []).append((fid, feat))
                step.append((tid,(x1,y1,x2,y2),cls))

                cv2.rectangle(fr,(x1,y1),(x2,y2),col,2)
                cv2.putText(fr,f'{lbl}:{tid}',(x1,max(y1-8,0)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,col,2)

                last_box[tid] = ((x1,y1,x2,y2), cls, TTL_FRAMES)   

        for tid in list(last_box):                                   
            bbox, cls, ttl = last_box[tid]
            if ttl == 0:
                last_box.pop(tid); continue
            last_box[tid] = (bbox, cls, ttl-1)
            x1,y1,x2,y2 = bbox
            col, lbl = COLOR_MAP[cls], LABEL_MAP[cls]
            cv2.rectangle(fr,(x1,y1),(x2,y2),col,2)
            cv2.putText(fr,f'{lbl}:{tid}',(x1,max(y1-8,0)),cv2.FONT_HERSHEY_SIMPLEX,0.7,col,2)         

        if save_tracks: frames[fid] = step
        out.write(fr)
        cv2.imshow(os.path.basename(src), fr)
        if cv2.waitKey(1)&0xFF==27: break
        fid += 1

    cap.release(); out.release(); cv2.destroyAllWindows()
    if save_tracks:
        with open("tacticam_tracks.pkl","wb") as f: pickle.dump(frames,f)
    return bank

WIN_SEC = 3        # Compare tracks only if med-frame diff ≤ WIN_SEC
THR = 0.55         # Cosine distance threshold

def robust_feat(vlist):
    # Median-pooled 1024-D descriptor from [(f,feat), …]
    return np.median(np.stack([v for _, v in vlist]), axis=0)

def hungarian_match(fb, ft, fps):
    ids_b, ids_t = list(fb), list(ft)
    D = np.full((len(ids_t), len(ids_b)), 1.0)      # init with large distance
    max_delta = WIN_SEC * fps

    desc_b = {i: robust_feat(fb[i]) for i in ids_b}
    fr_b   = {i: np.median([f for f,_ in fb[i]]) for i in ids_b}
    desc_t = {i: robust_feat(ft[i]) for i in ids_t}
    fr_t   = {i: np.median([f for f,_ in ft[i]]) for i in ids_t}

    for i,t in enumerate(ids_t):
        for j,b in enumerate(ids_b):
            if abs(fr_t[t] - fr_b[b]) > max_delta:
                continue                           # outside temporal window
            D[i,j] = cosine(desc_t[t], desc_b[b])

    from scipy.optimize import linear_sum_assignment
    rows, cols = linear_sum_assignment(D)
    mapping = {ids_t[r]: ids_b[c] for r,c in zip(rows, cols) if D[r,c] < THR}
    print(f"✔ kept {len(mapping)} / {len(ids_t)} tracks  (thr {THR}, win {WIN_SEC}s)")
    return mapping

#Repaint tacticam
def repaint(src,dst,map_,tracks_pkl):
    tracks = pickle.load(open(tracks_pkl,'rb'))
    cap=cv2.VideoCapture(src)
    w,h,fps=int(cap.get(3)),int(cap.get(4)),cap.get(cv2.CAP_PROP_FPS)
    out=cv2.VideoWriter(dst,cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
    fid=0
    while True:
        ok,fr=cap.read(); fid+=1
        if not ok: break
        for tid,(x1,y1,x2,y2),cls in tracks.get(fid-1,[]):
            bid=map_.get(tid,f'U{tid}')
            col=COLOR_MAP.get(cls,(255,255,255))
            lbl=LABEL_MAP[cls]
            cv2.rectangle(fr,(x1,y1),(x2,y2),col,2)
            cv2.putText(fr,f'{lbl}:{bid}',(x1,max(y1-8,0)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,col,2)
        out.write(fr); cv2.imshow("Remap",fr)
        if cv2.waitKey(1)&0xFF==27: break
    cap.release(); out.release(); cv2.destroyAllWindows()

# Pipeline
if __name__ == "__main__":
    print("▶ tracking broadcast …")
    fb = run_video("broadcast.mp4", tracker_b, "output_broadcast.mp4")

    print("▶ tracking tacticam …")
    ft = run_video("tacticam.mp4",  tracker_t, "output_tacticam.mp4",
                   save_tracks=True)

    print("▶ matching IDs …")
    fps = 30   # ← put your real FPS if different
    id_map = hungarian_match(fb, ft, fps=fps)
    for t,b in id_map.items(): print(f"Tacticam {t} → Broadcast {b}")

    print("▶ repainting tacticam …")
    repaint("tacticam.mp4", "remapped_tacticam.mp4",
            id_map, "tacticam_tracks.pkl")

    print(" Finished - check the three output videos.")


# Note: The script assumes the YOLOv11 model and StrongSORT weights are available in the specified paths.
# Make sure to adjust the paths and parameters according to your setup.
# The script also assumes the input videos are named "broadcast.mp4" and "tacticam.mp4".
# The output videos will be saved as "output_broadcast.mp4", "output_tacticam.mp4", and "remapped_tacticam.mp4".
# The tracks from the tacticam will be saved in "tacticam_tracks.pkl".  
