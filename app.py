import streamlit as st
import cv2
import numpy as np
import time

from PIL import Image
import io

st.set_page_config(page_title="🎨 多風格藝術濾鏡轉換系統", page_icon="🎨", layout="wide")

st.markdown("""
<style>
.title{text-align:center;font-size:2rem;font-weight:700}
.sub{text-align:center;color:#888;font-size:0.9rem;margin-bottom:1.5rem}
.timing{background:#1e3a5f;color:#60a5fa;padding:5px 14px;border-radius:20px;font-size:0.85rem}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="title">🎨 多風格藝術濾鏡轉換系統</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">第五組專題 · 指導教授：林皇辰 · 日系動漫 / 水彩 / 油畫 / 像素復古</div>', unsafe_allow_html=True)

@st.cache_resource




def analyze_color_tone(img_bgr):
    b_ch=np.mean(img_bgr[:,:,0]); g_ch=np.mean(img_bgr[:,:,1]); r_ch=np.mean(img_bgr[:,:,2])
    hsv=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2HSV).astype(np.float32)
    avg_v=np.mean(hsv[:,:,2]); warm=r_ch+g_ch*0.5-b_ch
    tone="warm" if warm>15 else ("cool" if warm<-15 else "neutral")
    bright="bright" if avg_v>170 else ("dark" if avg_v<80 else "normal")
    reason={"warm":"偏暖色","cool":"偏冷色","neutral":"中性色調"}[tone]
    return {"tone":tone,"brightness":bright,"reason":reason}

def apply_japanese_tone(result_bgr, tone_info):
    tone,brightness=tone_info["tone"],tone_info["brightness"]
    beta=20 if brightness=="dark" else 10
    result_bgr=cv2.convertScaleAbs(result_bgr,alpha=1.08,beta=beta)
    hsv=cv2.cvtColor(result_bgr,cv2.COLOR_BGR2HSV).astype(np.float32)
    if tone=="warm":
        hsv[:,:,1]=np.clip(hsv[:,:,1]*0.85,0,255); result_bgr=cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)
        b,g,r=cv2.split(result_bgr); r=np.clip(r.astype(np.int32)+8,0,255).astype(np.uint8); g=np.clip(g.astype(np.int32)+5,0,255).astype(np.uint8); b=np.clip(b.astype(np.int32)-5,0,255).astype(np.uint8)
    elif tone=="cool":
        hsv[:,:,1]=np.clip(hsv[:,:,1]*0.8,0,255); result_bgr=cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)
        b,g,r=cv2.split(result_bgr); r=np.clip(r.astype(np.int32)-8,0,255).astype(np.uint8); g=np.clip(g.astype(np.int32)+8,0,255).astype(np.uint8); b=np.clip(b.astype(np.int32)+20,0,255).astype(np.uint8)
    else:
        hsv[:,:,1]=np.clip(hsv[:,:,1]*0.9,0,255); result_bgr=cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)
        b,g,r=cv2.split(result_bgr); r=np.clip(r.astype(np.int32)+5,0,255).astype(np.uint8); g=np.clip(g.astype(np.int32)+8,0,255).astype(np.uint8); b=np.clip(b.astype(np.int32)+12,0,255).astype(np.uint8)
    result_bgr=cv2.merge([b,g,r]); glow=cv2.GaussianBlur(result_bgr,(0,0),10)
    return cv2.addWeighted(result_bgr,0.65,glow,0.35,0)

def anime_filter(img_bgr):
    # 純 OpenCV 動漫風：雙邊濾波 + 色彩量化 + 邊緣
    smooth = img_bgr.copy()
    for _ in range(2):
        smooth = cv2.bilateralFilter(smooth, d=9, sigmaColor=75, sigmaSpace=75)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(
        blur_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, blockSize=9, C=4
    )
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    data = smooth.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, 6, None, criteria, 4, cv2.KMEANS_RANDOM_CENTERS)
    quantized = centers[labels.flatten()].reshape(smooth.shape).astype(np.uint8)
    result = cv2.bitwise_and(quantized, edges_bgr)
    tone_info = analyze_color_tone(img_bgr)
    result = apply_japanese_tone(result, tone_info)
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    result = cv2.filter2D(result, -1, kernel)
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] = np.clip(hsv[:,:,1]*1.3, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def watercolor_filter(img_bgr):
    smooth=img_bgr.copy()
    for _ in range(3): smooth=cv2.bilateralFilter(smooth,d=9,sigmaColor=75,sigmaSpace=75)
    gray=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY); blur_gray=cv2.medianBlur(gray,7)
    edges=cv2.adaptiveThreshold(blur_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,blockSize=9,C=4)
    edges_bgr=cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR); result=cv2.bitwise_and(smooth,edges_bgr)
    h,w=result.shape[:2]; noise=np.random.randint(0,25,(h,w),dtype=np.uint8)
    noise_bgr=cv2.cvtColor(cv2.GaussianBlur(noise,(31,31),0),cv2.COLOR_GRAY2BGR)
    result=cv2.addWeighted(result,0.92,noise_bgr,0.08,0)
    tone=analyze_color_tone(img_bgr)["tone"]; hsv=cv2.cvtColor(result,cv2.COLOR_BGR2HSV).astype(np.float32)
    if tone=="warm":
        hsv[:,:,1]=np.clip(hsv[:,:,1]*1.15,0,255); result=cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)
        b,g,r=cv2.split(result); r=np.clip(r.astype(np.int32)+10,0,255).astype(np.uint8); g=np.clip(g.astype(np.int32)+5,0,255).astype(np.uint8); b=np.clip(b.astype(np.int32)-8,0,255).astype(np.uint8); result=cv2.merge([b,g,r])
    elif tone=="cool":
        hsv[:,:,1]=np.clip(hsv[:,:,1]*0.85,0,255); result=cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)
        b,g,r=cv2.split(result); r=np.clip(r.astype(np.int32)-10,0,255).astype(np.uint8); g=np.clip(g.astype(np.int32)+5,0,255).astype(np.uint8); b=np.clip(b.astype(np.int32)+18,0,255).astype(np.uint8); result=cv2.merge([b,g,r])
    else:
        hsv[:,:,1]=np.clip(hsv[:,:,1]*0.9,0,255); hsv[:,:,2]=np.clip(hsv[:,:,2]*1.05,0,255); result=cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)
        b,g,r=cv2.split(result); r=np.clip(r.astype(np.int32)+8,0,255).astype(np.uint8); g=np.clip(g.astype(np.int32)+10,0,255).astype(np.uint8); b=np.clip(b.astype(np.int32)+12,0,255).astype(np.uint8); result=cv2.merge([b,g,r])
    return cv2.GaussianBlur(result,(3,3),0)

def oil_painting_core(img_bgr,radius=4,levels=8):
    h,w=img_bgr.shape[:2]; result=np.zeros_like(img_bgr); gray=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
    for y in range(radius,h-radius):
        for x in range(radius,w-radius):
            patch=img_bgr[y-radius:y+radius+1,x-radius:x+radius+1]; gp=gray[y-radius:y+radius+1,x-radius:x+radius+1]
            bins=(gp.astype(np.int32)*levels)//256; counts=np.bincount(bins.ravel(),minlength=levels); mb=np.argmax(counts)
            result[y,x]=patch[bins==mb].mean(axis=0).astype(np.uint8)
    result[:radius,:]=img_bgr[:radius,:]; result[-radius:,:]=img_bgr[-radius:,:]; result[:,:radius]=img_bgr[:,:radius]; result[:,-radius:]=img_bgr[:,-radius:]
    return result

def oil_filter(img_bgr):
    h,w=img_bgr.shape[:2]; scale=min(1.0,400/max(h,w))
    small=cv2.resize(img_bgr,(int(w*scale),int(h*scale)),interpolation=cv2.INTER_AREA)
    result=cv2.resize(oil_painting_core(small),(w,h),interpolation=cv2.INTER_LANCZOS4)
    tone=analyze_color_tone(img_bgr)["tone"]; result=cv2.convertScaleAbs(result,alpha=1.12,beta=-10)
    hsv=cv2.cvtColor(result,cv2.COLOR_BGR2HSV).astype(np.float32)
    if tone=="warm":
        hsv[:,:,1]=np.clip(hsv[:,:,1]*1.3,0,255); result=cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)
        b,g,r=cv2.split(result); r=np.clip(r.astype(np.int32)+15,0,255).astype(np.uint8); g=np.clip(g.astype(np.int32)+5,0,255).astype(np.uint8); b=np.clip(b.astype(np.int32)-10,0,255).astype(np.uint8); result=cv2.merge([b,g,r])
    elif tone=="cool":
        hsv[:,:,1]=np.clip(hsv[:,:,1]*1.2,0,255); result=cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)
        b,g,r=cv2.split(result); r=np.clip(r.astype(np.int32)-5,0,255).astype(np.uint8); g=np.clip(g.astype(np.int32)+8,0,255).astype(np.uint8); b=np.clip(b.astype(np.int32)+15,0,255).astype(np.uint8); result=cv2.merge([b,g,r])
    else:
        hsv[:,:,1]=np.clip(hsv[:,:,1]*1.25,0,255); result=cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)
    kernel=np.array([[0,-0.5,0],[-0.5,3,-0.5],[0,-0.5,0]])
    return np.clip(cv2.filter2D(result,-1,kernel),0,255).astype(np.uint8)

RETRO_PALETTE=np.array([[0,0,0],[255,255,255],[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255],[255,0,255],[128,0,0],[0,128,0],[0,0,128],[255,128,0],[128,0,128],[0,128,128],[255,200,100],[100,200,255]],dtype=np.uint8)

def pixel_filter(img_bgr,pixel_size=8):
    h,w=img_bgr.shape[:2]; img_rgb=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
    sw,sh=max(1,w//pixel_size),max(1,h//pixel_size)
    pixelated=cv2.resize(cv2.resize(img_rgb,(sw,sh),cv2.INTER_AREA),(w,h),cv2.INTER_NEAREST)
    flat=pixelated.reshape(-1,3).astype(np.float32)
    diff=flat[:,np.newaxis,:]-RETRO_PALETTE.astype(np.float32)[np.newaxis,:,:]
    idx=np.argmin(np.sum(diff**2,axis=2),axis=1)
    result_bgr=cv2.cvtColor(RETRO_PALETTE[idx].reshape(h,w,3),cv2.COLOR_RGB2BGR)
    for y in range(0,h,pixel_size): cv2.line(result_bgr,(0,y),(w,y),(0,0,0),1)
    for x in range(0,w,pixel_size): cv2.line(result_bgr,(x,0),(x,h),(0,0,0),1)
    return result_bgr

def bgr2pil(img): return Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
def to_bytes(pil):
    buf=io.BytesIO(); pil.save(buf,format="JPEG",quality=92); return buf.getvalue()

# ── UI ──
col_l, col_r = st.columns([1,1])

with col_l:
    uploaded = st.file_uploader("📂 上傳圖片", type=["jpg","jpeg","png","webp"])
    style = st.radio("選擇風格", ["🌸 日系動漫","🖌️ 水彩風","🖼️ 油畫風","👾 像素復古"], horizontal=True)
    pixel_size = st.select_slider("像素大小（像素復古專用）", options=[4,6,8,10,12,16,20,24], value=8) if style=="👾 像素復古" else 8
    run_btn = st.button("🎨 套用濾鏡", type="primary", use_container_width=True)

with col_r:
    if uploaded is None:
        st.info("請先上傳一張圖片，再選擇風格套用")
    else:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        h,w = img_bgr.shape[:2]
        if w > 512:
            img_bgr = cv2.resize(img_bgr,(512,int(h*512/w)),interpolation=cv2.INTER_AREA)

        if run_btn:
            with st.spinner("處理中，請稍候..."):
                t0=time.time()
                if   style=="🌸 日系動漫": result=anime_filter(img_bgr)
                elif style=="🖌️ 水彩風":   result=watercolor_filter(img_bgr)
                elif style=="🖼️ 油畫風":   result=oil_filter(img_bgr)
                else:                       result=pixel_filter(img_bgr,pixel_size)
                elapsed=time.time()-t0

            tone_info=analyze_color_tone(img_bgr)
            st.markdown(f'<span class="timing">⏱ {elapsed:.2f} 秒　｜　{tone_info["reason"]}</span>', unsafe_allow_html=True)
            st.markdown(" ")
            c1,c2=st.columns(2)
            with c1: st.caption("原圖"); st.image(bgr2pil(img_bgr),use_column_width=True)
            with c2: st.caption(style); st.image(bgr2pil(result),use_column_width=True)
            fname=uploaded.name.rsplit(".",1)[0]
            st.download_button("⬇ 下載結果",data=to_bytes(bgr2pil(result)),file_name=f"{style.split()[1]}_{fname}.jpg",mime="image/jpeg",use_container_width=True)
        else:
            st.caption("原圖預覽")
            st.image(bgr2pil(img_bgr), use_column_width=True)

st.markdown("---")
st.caption("圖片於伺服器本地處理 · 不會對外傳送")
