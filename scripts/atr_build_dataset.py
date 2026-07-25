"""Build the NVESD ATR MWIR detection subset (COCO, roboflow layout) from ARF+AGT.
Frame-sampled, temporal train/val split, geometric box synthesis. Boxes carry
range/aspect/center; images carry scenario/day-night/nominal-range."""
from __future__ import annotations
import re, struct, math, json, sys
from pathlib import Path
import numpy as np
from PIL import Image

TGT_DIMS = {"PICKUP":(5.41,1.80,1.68),"SUV":(4.57,1.73,1.73),"BTR70":(7.62,2.79,2.16),
 "BRDM2":(5.72,2.29,2.03),"BMP2":(6.73,3.15,2.45),"T72":(6.59,3.59,2.21),
 "ZSU23-4":(5.94,2.88,2.34),"2S3":(6.71,3.23,2.72),"MTLB":(6.25,3.10,2.30),
 "D20":(8.35,2.40,1.80),"MAN":(0.6,0.6,1.8)}
FOV_H,FOV_V=3.4,2.6
SRC=Path("/Volumes/T7 Shield/ATR_Database/ATR Database/cegr")
OUT=Path("/Volumes/T7 Shield/ATR_Database/prepared/nvesd_atr_cegr")
N_PER_LOOK=15; VAL_FRAC=0.3

# scenario -> (nominal_range_m, day/night)  (User Ref Guide Tables 2 & 3)
RANGE_TABLE={}
for scen,rng in zip([2003,2005,2007,2009,2011,2013,2015,2017,2019],range(1000,5001,500)): RANGE_TABLE[scen]=(rng,"day")
for scen,rng in zip([1923,1925,1927,1929,1931,1933,1935,1937,1939],range(1000,5001,500)): RANGE_TABLE[scen]=(rng,"night")
for scen,rng in zip([2002,2004,2006,2008,2010,2012],range(500,3001,500)): RANGE_TABLE[scen]=(rng,"day")
for scen,rng in zip([1926,1928,1932,1934,1936,1938],range(500,3001,500)): RANGE_TABLE[scen]=(rng,"night")

class ARF:
    def __init__(s,p):
        s.f=open(p,"rb"); h=s.f.read(32)
        s.magic,s.ver,s.rows,s.cols,s.itype,s.n,s.off,_=struct.unpack(">8I",h)
        s.px=2; s.fb=s.rows*s.cols*s.px
    def frame(s,i):
        s.f.seek(s.off+i*s.fb); b=s.f.read(s.fb)
        return np.frombuffer(b,">u2").astype(np.float32).reshape(s.rows,s.cols)
    def close(s): s.f.close()

def norm8(fr):
    a,b=np.percentile(fr,[1,99]); b=b if b>a else a+1
    return np.clip((fr-a)/(b-a)*255,0,255).astype(np.uint8)

def parse_agt(p):
    txt=Path(p).read_text(errors="ignore"); i=txt.find("TgtSect")
    txt=txt[i:] if i>=0 else txt; out={}
    for m in re.finditer(r'TgtUpd\s*\{(.*?)\n\s*\}\s*(?=TgtUpd|\Z)',txt,re.S):
        blk=m.group(1); fm=re.search(r'Frame#\s*(\d+)',blk)
        if not fm: continue
        f=int(fm.group(1))-1; tg=[]
        for tm in re.finditer(r'Tgt\s*\{(.*?)\n\s*\}',blk,re.S):
            t=tm.group(1); tt=re.search(r'TgtType\s+"([^"]+)"',t); pl=re.search(r'PixLoc\s+(-?\d+)\s+(-?\d+)',t)
            rg=re.search(r'\n\s*Range\s+([\d.]+)',t); asp=re.search(r'Aspect\s+(-?[\d.]+)',t)
            if not(tt and pl): continue
            if tt.group(1).startswith("BB"): continue   # blackbody calibration
            tg.append(dict(tt=tt.group(1),x=int(pl.group(1)),y=int(pl.group(2)),
                rng=float(rg.group(1)) if rg else None,asp=float(asp.group(1)) if asp else 0.0))
        if tg: out[f]=tg
    return out

def box(t,rows,cols):
    d=TGT_DIMS.get(t["tt"]); 
    if not d: return None
    L,W,H=d; rng=t["rng"] or 1e9; a=math.radians(t["asp"] or 0)
    pw=(abs(L*math.sin(a))+abs(W*math.cos(a)))/rng/(math.radians(FOV_H)/cols)
    ph=H/rng/(math.radians(FOV_V)/rows)
    pw=max(pw,1.); ph=max(ph,1.)
    return [t["x"]-pw/2,t["y"]-ph/2,pw,ph]

def main():
    looks=sorted([p for p in (SRC/"agt").glob("cegr*_0*.agt") if not p.name.startswith("._")])
    cats={}; 
    def cid(name): return cats.setdefault(name,len(cats)+1)
    split={"train":dict(images=[],annotations=[]),"valid":dict(images=[],annotations=[])}
    for sp in split: (OUT/sp).mkdir(parents=True,exist_ok=True)
    img_id={"train":0,"valid":0}; ann_id={"train":0,"valid":0}; n_looks=0
    for agt in looks:
        scen=int(agt.name[4:9]); 
        if scen not in RANGE_TABLE: continue   # keep only the target range-study scenarios
        arf=SRC/"arf"/f"{agt.stem}.arf"
        if not arf.exists(): continue
        gt=parse_agt(agt)
        if not gt: continue
        nom,dn=RANGE_TABLE[scen]
        r=ARF(arf); frames_with_gt=[f for f in sorted(gt) if 0<=f<r.n]
        if not frames_with_gt: r.close(); continue
        # sample N evenly across frames that have GT
        idx=frames_with_gt if len(frames_with_gt)<=N_PER_LOOK else \
            [frames_with_gt[round(i*(len(frames_with_gt)-1)/(N_PER_LOOK-1))] for i in range(N_PER_LOOK)]
        idx=sorted(set(idx)); ncut=int(len(idx)*(1-VAL_FRAC))
        for k,fi in enumerate(idx):
            sp="train" if k<ncut else "valid"   # temporal split
            arr=norm8(r.frame(fi)); 
            fn=f"{agt.stem}_f{fi:04d}.png"
            Image.fromarray(arr).convert("RGB").save(OUT/sp/fn)
            img_id[sp]+=1; iid=img_id[sp]
            split[sp]["images"].append(dict(id=iid,file_name=fn,width=r.cols,height=r.rows,
                scenario=scen,sensor="cegr",day_night=dn,nominal_range_m=nom))
            for t in gt[fi]:
                b=box(t,r.rows,r.cols)
                if not b: continue
                ann_id[sp]+=1
                split[sp]["annotations"].append(dict(id=ann_id[sp],image_id=iid,
                    category_id=cid(t["tt"]),bbox=[float(v) for v in b],
                    area=float(b[2]*b[3]),iscrowd=0,
                    center=[t["x"],t["y"]],range_m=t["rng"],aspect=t["asp"]))
        r.close(); n_looks+=1
        if n_looks%10==0: print(f"  {n_looks} looks done...",flush=True)
    catlist=[dict(id=i,name=n) for n,i in cats.items()]
    for sp in split:
        split[sp]["categories"]=catlist
        split[sp]["info"]={"description":"NVESD ATR MWIR (cegr) detection subset"}
        json.dump(split[sp],open(OUT/sp/"_annotations.coco.json","w"))
    print(f"DONE: {n_looks} looks | train {img_id['train']} imgs/{ann_id['train']} ann | "
          f"valid {img_id['valid']} imgs/{ann_id['valid']} ann | classes={list(cats)}")

if __name__=="__main__": main()
