"""NVESD ATR prep: ARF (16-bit MWIR) + AGT (center-point GT) -> PNG + COCO.
Boxes are SYNTHESIZED from target dimensions + range + aspect + sensor IFOV,
since AGT gives only a center point (PixLoc)."""
from __future__ import annotations
import re, struct, math
from pathlib import Path
import numpy as np

# target L/W/H (m) from Metadata/Target Data/target_data.xls
TGT_DIMS = {
 "PICKUP":(5.41,1.80,1.68),"SUV":(4.57,1.73,1.73),"BTR70":(7.62,2.79,2.16),
 "BRDM2":(5.72,2.29,2.03),"BMP2":(6.73,3.15,2.45),"T72":(6.59,3.59,2.21),
 "ZSU23-4":(5.94,2.88,2.34),"2S3":(6.71,3.23,2.72),"MTLB":(6.25,3.10,2.30),
 "D20":(8.35,2.40,1.80),
}
HUMAN_DIMS=(0.6,0.6,1.8)  # standing person approx (L,W,H)
FOV_H_DEG, FOV_W_DEG = 3.4, 2.6  # horizontal(cols), vertical(rows) per User Guide

def read_arf(path):
    with open(path,"rb") as f: raw=f.read()
    magic,ver,rows,cols,itype,nframes,off,flags = struct.unpack(">8I", raw[:32])
    assert magic==3149642413 and ver==2, (magic,ver)
    px = {0:1,1:2,2:2,3:2,4:4,5:2,6:4}.get(itype,2)
    dt = {1:">u2",5:">u2",2:">u2",0:">u1",3:">i2"}.get(itype,">u2")
    n = rows*cols
    frames=[]
    for i in range(nframes):
        s=off+i*n*px
        fr=np.frombuffer(raw[s:s+n*px],dtype=dt).astype(np.float32).reshape(rows,cols)
        frames.append(fr)
    return frames, rows, cols

def norm8(fr, lo=1.0, hi=99.0):
    a,b=np.percentile(fr,[lo,hi])
    if b<=a: b=a+1
    return np.clip((fr-a)/(b-a)*255,0,255).astype(np.uint8)

def parse_agt(path):
    """Return {frame_idx(0-based): [ {tgt_type,x,y,range,aspect}, ... ]}."""
    txt=Path(path).read_text(errors="ignore")
    # split into TgtSect -> TgtUpd blocks
    i=txt.find("TgtSect")
    txt=txt[i:] if i>=0 else txt
    out={}
    # iterate TgtUpd blocks
    for m in re.finditer(r'TgtUpd\s*\{(.*?)\n\s*\}\s*(?=TgtUpd|\Z)', txt, re.S):
        blk=m.group(1)
        fm=re.search(r'Frame#\s*(\d+)', blk)
        if not fm: continue
        fidx=int(fm.group(1))-1
        tgts=[]
        for tm in re.finditer(r'Tgt\s*\{(.*?)\n\s*\}', blk, re.S):
            t=tm.group(1)
            tt=re.search(r'TgtType\s+"([^"]+)"',t)
            pl=re.search(r'PixLoc\s+(-?\d+)\s+(-?\d+)',t)
            rg=re.search(r'\n\s*Range\s+([\d.]+)',t)
            asp=re.search(r'Aspect\s+(-?[\d.]+)',t)
            if not(tt and pl): continue
            tgts.append(dict(tgt_type=tt.group(1),x=int(pl.group(1)),y=int(pl.group(2)),
                             range=float(rg.group(1)) if rg else None,
                             aspect=float(asp.group(1)) if asp else 0.0))
        if tgts: out[fidx]=tgts
    return out

def synth_box(t, rows, cols):
    """Axis-aligned px box centered at PixLoc from dims+range+aspect."""
    dims = TGT_DIMS.get(t["tgt_type"], HUMAN_DIMS if "H" in t["tgt_type"].upper() or t["range"] else HUMAN_DIMS)
    L,W,H = dims
    rng = t["range"] or 1e9
    asp = math.radians(t["aspect"] or 0.0)
    proj_w = abs(L*math.sin(asp)) + abs(W*math.cos(asp))   # horizontal silhouette
    ifov_h = math.radians(FOV_H_DEG)/cols   # rad/px horizontal
    ifov_v = math.radians(FOV_W_DEG)/rows
    pw = proj_w/rng/ifov_h
    ph = H/rng/ifov_v
    pw=max(pw,1.0); ph=max(ph,1.0)
    x1=t["x"]-pw/2; y1=t["y"]-ph/2
    return [float(x1),float(y1),float(pw),float(ph)], (pw,ph)

if __name__=="__main__":
    import sys
    base=Path("/Volumes/T7 Shield/ATR_Database/ATR Database/cegr")
    stem=sys.argv[1] if len(sys.argv)>1 else "cegr02003_0001"
    frames,rows,cols=read_arf(base/"arf"/f"{stem}.arf")
    gt=parse_agt(base/"agt"/f"{stem}.agt")
    print(f"{stem}: {len(frames)} frames {rows}x{cols}; GT frames={sorted(gt)[:5]}...")
    # pick first frame with a target
    fi=sorted(gt)[0]
    print(f"frame {fi} targets:")
    from PIL import Image, ImageDraw
    img=Image.fromarray(norm8(frames[fi])).convert("RGB")
    dr=ImageDraw.Draw(img)
    for t in gt[fi]:
        box,(pw,ph)=synth_box(t,rows,cols)
        print(f"  {t['tgt_type']:8} range={t['range']}m aspect={t['aspect']:.0f} "
              f"center=({t['x']},{t['y']}) -> box {pw:.1f}x{ph:.1f}px")
        x1,y1,w,h=box
        dr.rectangle([x1,y1,x1+w,y1+h],outline=(255,0,0),width=1)
        dr.point([t['x'],t['y']],fill=(0,255,0))
    out=f"/tmp/atr/overlay_{stem}.png"
    img.resize((cols*2,rows*2)).save(out)
    print("saved",out)
