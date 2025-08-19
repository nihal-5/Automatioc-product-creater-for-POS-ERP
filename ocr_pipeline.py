#!/usr/bin/env python3
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFilter
from datetime import datetime, UTC
import os, sys, json, re, time

# ---------------- Paths & Config ----------------
BASE = Path(__file__).resolve().parent
IN_DIR = BASE / "input"
OUT_DIR = BASE / "output"
CROPS_DIR = OUT_DIR / "crops"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)

MAX_SIDE    = int(os.getenv("MAX_SIDE","2000"))
YOLO_MODEL  = os.getenv("YOLO_MODEL","yolov8m-world.pt")
YOLO_CONF   = float(os.getenv("YOLO_CONF","0.25"))
YOLO_IOU    = float(os.getenv("YOLO_IOU","0.45"))
YOLO_CLASSES = [c.strip() for c in os.getenv(
    "YOLO_CLASSES",
    "product,package,box,bottle,can,jar,bag,label,sticker,seal,logo,brand,brand name,"
    "nutrition facts,nutrition label,nutrition panel,table,ingredients,barcode,qr code"
).split(",") if c.strip()]
OCR_LANGS   = [s.strip() for s in os.getenv("OCR_LANGS","en").split(",") if s.strip()]

# Hero image export size (square)
HERO_SIZE   = int(os.getenv("HERO_SIZE","1024"))

# Cutout (GrabCut) tuning via env, without touching pipeline
CUT_PAD_PCT = float(os.getenv("CUT_PAD_PCT", "0.06"))  # padding around best box before grabcut
CUT_ITERS   = int(os.getenv("CUT_ITERS", "5"))         # grabcut iterations

# ---------------- Helpers ----------------
def pick_single_image():
    if not IN_DIR.exists():
        sys.exit("Missing input/ folder")
    imgs=[p for p in IN_DIR.iterdir()
          if p.is_file() and p.suffix.lower() in (".jpg",".jpeg",".png",".webp",".tif",".tiff",".bmp")]
    if len(imgs)==0: sys.exit("No image in input/. Put exactly one image.")
    if len(imgs)>1: sys.exit(f"Found {len(imgs)} images. Keep only one.")
    return imgs[0]

def resize_limit(pil, max_side):
    w,h=pil.size; ms=max(w,h)
    if ms<=max_side: return pil,1.0
    s=max_side/ms
    return pil.resize((int(w*s),int(h*s)), Image.LANCZOS), s

def upscale_back(items, scale):
    if scale==1.0 or not items: return
    s=1.0/scale
    for it in items:
        b=it["bbox"]; b["x"]=int(b["x"]*s); b["y"]=int(b["y"]*s); b["w"]=int(b["w"]*s); b["h"]=int(b["h"]*s)

def clamp_box(x,y,w,h,W,H):
    x=max(0,min(x,W-1)); y=max(0,min(y,H-1))
    w=max(1,min(w,W-x)); h=max(1,min(h,H-y))
    return x,y,w,h

# ---------------- Drawing ----------------
def draw_boxes(pil, dets, out_path, color=(255,0,0), width=3):
    img=pil.copy().convert("RGB"); d=ImageDraw.Draw(img)
    try: font=ImageFont.load_default()
    except: font=None
    for i,det in enumerate(dets,1):
        b=det["bbox"]; x,y,w,h=b["x"],b["y"],b["w"],b["h"]
        d.rectangle([x,y,x+w,y+h], outline=color, width=width)
        label=f"{i}:{det.get('cls','')}:{int(det.get('conf',0)*100)}"
        if font: tw,th=d.textbbox((0,0),label,font=font)[2:]
        else: tw,th=(len(label)*6,10)
        tx,ty=x+1,max(0,y-th-6)
        d.rectangle([tx,ty,tx+tw+6,ty+th+4], fill=color)
        d.text((tx+3,ty+2), label, fill=(255,255,255), font=font)
    img.save(out_path, quality=95)
    return out_path

def draw_lines(pil, lines, out_path, color=(255,0,0), width=2):
    img=pil.copy().convert("RGB"); d=ImageDraw.Draw(img)
    try: font=ImageFont.load_default()
    except: font=None
    for it in lines:
        b=it["bbox"]; x,y,w,h=b["x"],b["y"],b["w"],b["h"]
        d.rectangle([x,y,x+w,y+h], outline=color, width=width)
        label=f"{it['id']}:{int(it.get('conf',0)*100)}"
        if font: tw,th=d.textbbox((0,0),label,font=font)[2:]
        else: tw,th=(len(label)*6,10)
        tx,ty=x+1,max(0,y-th-6)
        d.rectangle([tx,ty,tx+tw+6,ty+th+4], fill=color)
        d.text((tx+3,ty+2), label, fill=(255,255,255), font=font)
    img.save(out_path, quality=95)
    return out_path

# ---------------- Mask outside text clusters (unchanged feature) ----------------
def build_text_mask(im, boxes, pad=4, dilate=7):
    W,H = im.size
    mask = Image.new("L",(W,H),0)
    d = ImageDraw.Draw(mask)
    for b in boxes:
        x,y,w,h=b["x"],b["y"],b["w"],b["h"]
        x,y,w,h = clamp_box(x-pad,y-pad,w+2*pad,h+2*pad,W,H)
        d.rectangle([x,y,x+w,y+h], fill=255)
    if dilate and dilate>1:
        mask = mask.filter(ImageFilter.MaxFilter(size=(dilate//2)*2+1))
    return mask

def apply_mask_keep_text(im, text_mask, bg_color=(255,255,255)):
    W,H = im.size
    bg = Image.new("RGB",(W,H),bg_color)
    if text_mask.mode!="L":
        text_mask = text_mask.convert("L")
    return Image.composite(im, bg, text_mask)

# ---------------- Models ----------------
def run_yolo_world(pil):
    try:
        from ultralytics import YOLOWorld
    except Exception as e:
        return [], f"ultralytics not available: {e}"
    m=YOLOWorld(YOLO_MODEL)
    m.set_classes(YOLO_CLASSES)
    res=m.predict(pil, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)
    dets=[]
    if res:
        r=res[0]; names=r.names if hasattr(r,"names") and isinstance(r.names,dict) else {}
        if getattr(r,"boxes",None) is not None:
            for b in r.boxes:
                x0,y0,x1,y1=[int(v) for v in b.xyxy[0].tolist()]
                cls_id=int(b.cls[0].item()) if b.cls is not None else -1
                name=names.get(cls_id,str(cls_id))
                conf=float(b.conf[0].item()) if b.conf is not None else 0.0
                dets.append({"cls":name,"conf":conf,"bbox":{"x":x0,"y":y0,"w":x1-x0,"h":y1-y0}})
    return dets, None

def run_easyocr(pil):
    try:
        import easyocr, numpy as np
    except Exception as e:
        return [], f"easyocr not available: {e}"
    reader=easyocr.Reader(OCR_LANGS, gpu=False, verbose=False)
    arr=ImageOps.exif_transpose(pil).convert("RGB"); arr=np.array(arr)
    res=reader.readtext(arr, detail=1, paragraph=False)
    lines=[]
    for i,(box,text,conf) in enumerate(res,1):
        xs=[int(p[0]) for p in box]; ys=[int(p[1]) for p in box]
        x0,y0,x1,y1=min(xs),min(ys),max(xs),max(ys)
        lines.append({"id":i,"text":str(text),"conf":float(conf),"bbox":{"x":x0,"y":y0,"w":x1-x0,"h":y1-y0}})
    return lines, None

def decode_barcodes(pil):
    try:
        from pyzbar.pyzbar import decode, ZBarSymbol
    except Exception as e:
        return {"note":f"pyzbar not available: {e}","items":[]}
    gray=ImageOps.exif_transpose(pil).convert("L")
    out=decode(gray, symbols=[ZBarSymbol.EAN13,ZBarSymbol.EAN8,ZBarSymbol.CODE128,ZBarSymbol.QRCODE,ZBarSymbol.UPCA,ZBarSymbol.UPCE])
    items=[]
    for it in out:
        x,y,w,h=it.rect.left,it.rect.top,it.rect.width,it.rect.height
        items.append({"type":it.type,"data":it.data.decode("utf-8","ignore"),
                      "bbox":{"x":x,"y":y,"w":w,"h":h}})
    return {"note":None,"items":items}

# ---------------- Summary extraction (unchanged) ----------------
def find_best(lines, W, H):
    text_all=" ".join(l["text"] for l in lines)
    expiry=None
    m=re.search(r"(best\s*before|use\s*by|expiry|exp\.?|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[^\n]{0,25}(\d{1,2}[\-/\s]\d{1,2}[\-/\s]\d{2,4}|\d{4}|\d{1,2}\s*[A-Za-z]{3,}\s*\d{2,4})", text_all, re.I)
    if m: expiry=m.group(0)
    brand=None
    for l in lines:
        t=l["text"].strip()
        if re.search(r"(distributed\s+by|brand|company|inc\.|llc|ltd)", t, re.I):
            brand=t; break
    if brand is None:
        top=[l for l in lines if l["bbox"]["y"]<H*0.25 and len(l["text"])<=30]
        if top: brand=sorted(top, key=lambda k: -len(k["text"]))[0]["text"]
    product=None
    mids=[l for l in lines if l["bbox"]["y"]<H*0.35 and not re.search(r"nutrition|facts|ingredients|calories", l["text"], re.I)]
    if mids: product=sorted(mids, key=lambda k:(k["bbox"]["y"],-len(k["text"])))[0]["text"]
    weight=None
    mw=re.search(r"(\d+(\.\d+)?\s*(g|kg|oz|lb|ml|l))", text_all, re.I)
    if mw: weight=mw.group(0)
    serving=None
    ms=re.search(r"(serving\s*size|per\s*serving)[^\n]{0,20}", text_all, re.I)
    if ms: serving=ms.group(0)
    return {"brand":brand,"product_name":product,"expiry":expiry,"net_weight":weight,"serving_size":serving}

# ---------------- Hero helpers ----------------
_PRODUCT_LIKE = {"product","package","box","bottle","can","jar","bag","label"}

def _pick_best_product_box(dets):
    if not dets: return None
    scored=[]
    for d in dets:
        name=(d.get("cls") or "").lower()
        is_prod = 1 if any(k in name for k in _PRODUCT_LIKE) else 0
        b=d["bbox"]; area=b["w"]*b["h"]
        scored.append((is_prod, d.get("conf",0.0), area, d))
    scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
    return scored[0][3] if scored else None

def _square_from_rect(x,y,w,h,W,H,scale=1.07):
    cx,cy = x + w/2, y + h/2
    side = int(max(w,h)*scale)
    sx = int(cx - side/2); sy = int(cy - side/2)
    return clamp_box(sx,sy,side,side,W,H)

def _make_hero_from_box(pil, det, size=HERO_SIZE, white=(255,255,255)):
    W,H = pil.size
    b = det["bbox"]; x,y,w,h = b["x"], b["y"], b["w"], b["h"]
    sx,sy,sw,sh = _square_from_rect(x,y,w,h,W,H,scale=1.07)
    crop = pil.crop((sx,sy,sx+sw,sy+sh))
    canvas = Image.new("RGB",(size,size),white)
    canvas.paste(crop.resize((size,size), Image.LANCZOS), (0,0))
    return canvas

# ---- Tight cutout on white using GrabCut + largest component cleanup ----
def _cutout_on_white(pil, det, size=HERO_SIZE, white=(255,255,255)):
    try:
        import numpy as np, cv2
    except Exception as e:
        return None, f"cutout skipped (opencv not available: {e})"

    W, H = pil.size
    b = det["bbox"]; x,y,w,h = b["x"], b["y"], b["w"], b["h"]
    pad = int(CUT_PAD_PCT * max(w,h))
    rx, ry = max(0, x-pad), max(0, y-pad)
    rw, rh = min(W-rx, w+2*pad), min(H-ry, h+2*pad)

    img = ImageOps.exif_transpose(pil).convert("RGB")
    img_np = np.array(img)
    mask = np.zeros(img_np.shape[:2], np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    cv2.grabCut(img_np, mask, (rx, ry, rw, rh), bgdModel, fgdModel, CUT_ITERS, cv2.GC_INIT_WITH_RECT)
    gc = np.where((mask==cv2.GC_FGD) | (mask==cv2.GC_PR_FGD), 255, 0).astype("uint8")

    # Keep only the largest connected component to drop stray bits
    num, labels, stats, _ = cv2.connectedComponentsWithStats(gc, connectivity=8)
    if num <= 1:
        return None, "cutout skipped (mask empty)"
    # ignore label 0 (background), pick largest by area
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    gc = np.where(labels==largest, 255, 0).astype("uint8")

    gc = cv2.medianBlur(gc, 5)

    fg = img_np.copy()
    white_bg = 255*np.ones_like(fg, dtype=np.uint8)
    cut = np.where(gc[...,None]==255, fg, white_bg)

    ys, xs = np.where(gc==255)
    x0,x1 = xs.min(), xs.max()
    y0,y1 = ys.min(), ys.max()
    side = max(x1-x0+1, y1-y0+1)
    cx = (x0+x1)//2; cy=(y0+y1)//2
    sx = max(0, min(W-side, cx - side//2))
    sy = max(0, min(H-side, cy - side//2))

    cut_pil = Image.fromarray(cut).crop((sx,sy,sx+side,sy+side)).resize((size,size), Image.LANCZOS)
    return cut_pil, None

# ---------------- Main ----------------
def main():
    img_path=pick_single_image()
    print(f"image={img_path}")
    orig=Image.open(img_path); orig=ImageOps.exif_transpose(orig).convert("RGB")
    proc,scale=resize_limit(orig, MAX_SIDE)
    if scale!=1.0: print(f"resized_to={proc.size[0]}x{proc.size[1]} max_side={MAX_SIDE}")

    # 1) YOLO
    t0=time.time(); dets, yolo_note = run_yolo_world(proc); t1=time.time()

    # 2) OCR (as word/line boxes)
    lines, ocr_note = run_easyocr(proc); t2=time.time()

    # 3) Barcode
    bc=decode_barcodes(proc); t3=time.time()

    # upscale back to original coords if needed
    if scale!=1.0:
        upscale_back(dets,scale)
        upscale_back(lines,scale)

    # ---- Save overlays (unchanged core outputs) ----
    det_img = OUT_DIR / f"{img_path.stem}_detect.jpeg"
    ocr_img = OUT_DIR / f"{img_path.stem}_ocr.jpeg"
    draw_boxes(orig, dets, det_img)
    draw_lines(orig, lines, ocr_img)

    # ---- Masked image: white outside text clusters (unchanged feature) ----
    text_boxes = [l["bbox"] for l in lines] if lines else []
    if text_boxes:
        mask = build_text_mask(orig, text_boxes, pad=4, dilate=7)
        masked = apply_mask_keep_text(orig, mask, bg_color=(255,255,255))
        masked_img = OUT_DIR / f"{img_path.stem}_masked.jpeg"
        masked.save(masked_img, quality=95)
    else:
        masked_img = None

    # 4) Square hero (existing)
    hero_path = None
    best_box = _pick_best_product_box(dets)
    if best_box is not None:
        hero = _make_hero_from_box(orig, best_box, size=HERO_SIZE)
        hero_path = OUT_DIR / f"{img_path.stem}_hero.jpeg"
        hero.save(hero_path, quality=95)

    # 5) NEW: tight hero cutout on pure white (separate pipeline)
    hero_cutout_path = None
    cut_note = None
    if best_box is not None:
        cut_img, cut_note = _cutout_on_white(orig, best_box, size=HERO_SIZE)
        if cut_img is not None:
            hero_cutout_path = OUT_DIR / f"{img_path.stem}_hero_cutout.jpeg"
            cut_img.save(hero_cutout_path, quality=95)

    # ---- Summary & JSON (unchanged) ----
    H=orig.size[1]; W=orig.size[0]
    summary=find_best(lines,W,H)
    avg_conf=round(sum(l["conf"] for l in lines)/len(lines),4) if lines else 0.0

    data={
        "file": img_path.name,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00","Z"),
        "models":{"detector":YOLO_MODEL,"ocr":"easyocr"},
        "timings":{"yolo_s":round(t1-t0,3),"easyocr_s":round(t2-t1,3),"barcode_s":round(t3-t2,3)},
        "detections":{"yolo_world":dets,"classes":YOLO_CLASSES,"yolo_note":yolo_note},
        "ocr":{"line_count":len(lines),"avg_conf":avg_conf,"lines":lines,"ocr_note":ocr_note},
        "barcode":bc,
        "summary":summary,
        "outputs":{
            "detect_image": str(det_img.resolve()),
            "ocr_image": str(ocr_img.resolve()),
            "masked_image": str(masked_img.resolve()) if masked_img else None,
            "hero_image": str(hero_path.resolve()) if hero_path else None,
            "hero_cutout_image": str(hero_cutout_path.resolve()) if hero_cutout_path else None
        },
        "notes":{"cutout": cut_note}
    }
    out_json = OUT_DIR / f"{img_path.stem}.json"
    with out_json.open("w",encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Console: absolute paths so you can find files easily
    print(f"detect_image={det_img.resolve()}")
    print(f"ocr_image={ocr_img.resolve()}")
    if masked_img: print(f"masked_image={masked_img.resolve()}")
    else: print("masked_image=None (no OCR boxes)")
    if hero_path: print(f"hero_image={hero_path.resolve()}")
    else: print("hero_image=None (no suitable product/package box)")
    if hero_cutout_path: print(f"hero_cutout_image={hero_cutout_path.resolve()}")
    else: print("hero_cutout_image=None (opencv not installed or mask empty)")
    print(f"json={out_json.resolve()}")
    if yolo_note: print(f"note={yolo_note}")
    if ocr_note: print(f"note={ocr_note}")
    if bc.get("note"): print(f"barcode_note={bc['note']}")
    if cut_note: print(f"cutout_note={cut_note}")

if __name__=="__main__":
    main()
