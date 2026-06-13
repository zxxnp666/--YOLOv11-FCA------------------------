import random
from io import BytesIO

import cv2
import numpy as np
import streamlit as st
from PIL import Image


def add_random_haze(
    img_bgr: np.ndarray,
    beta_range=(0.6, 2.0),
    A_range=(0.7, 1.0),
) -> tuple[np.ndarray, dict]:
    """
    给 BGR 图像添加随机雾气（基于简化大气散射模型）

    参数:
        img_bgr: 原始图像 (H, W, 3)，BGR，uint8
        beta_range: 雾浓度系数 beta 的随机范围，越大雾越浓
        A_range: 大气光强度 A 的随机范围，[0,1]，越靠近 1 越偏白
    """
    h, w, _ = img_bgr.shape

    # 1. 随机采样雾参数
    beta = random.uniform(*beta_range)
    A = random.uniform(*A_range)

    # 2. 生成平滑随机“深度图” d(x) ∈ [0,1]
    #    为了避免大图高斯模糊过慢：先在低分辨率生成，再上采样
    max_dim = 512
    scale = max(1, int(max(h, w) / max_dim))
    small_h = max(1, h // scale)
    small_w = max(1, w // scale)

    noise_small = np.random.rand(small_h, small_w).astype(np.float32)
    sigma = max(small_h, small_w) / 10
    noise_small = cv2.GaussianBlur(noise_small, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    noise = cv2.resize(noise_small, (w, h), interpolation=cv2.INTER_CUBIC)
    noise -= noise.min()
    noise /= (noise.max() + 1e-8)
    d = noise

    # 3. 透射率 t(x) = exp(-beta * d(x))
    t = np.exp(-beta * d)
    t = t[..., None]

    # 4. 大气散射模型 I = J * t + A * (1 - t)
    J = img_bgr.astype(np.float32) / 255.0
    A_vec = np.array([A, A, A], dtype=np.float32)
    I = J * t + A_vec * (1.0 - t)

    hazy_bgr = np.clip(I * 255.0, 0, 255).astype(np.uint8)

    info = {
        "beta_雾浓度系数": round(beta, 3),
        "A_大气光强度": round(A, 3),
    }
    return hazy_bgr, info


def main():
    st.set_page_config(
        page_title="随机雾气生成小工具",
        page_icon="🌫️",
        layout="wide",
    )

    st.title("随机雾气生成小工具")
    st.markdown("上传一张图片，点击按钮即可随机生成不同雾气浓度、分布的雾天效果。")

    st.sidebar.header("雾气参数范围（可选调整）")
    beta_min, beta_max = st.sidebar.slider(
        "雾浓度系数 β 范围（越大雾越浓）",
        min_value=0.1,
        max_value=3.0,
        value=(0.6, 2.0),
        step=0.1,
    )
    A_min, A_max = st.sidebar.slider(
        "大气光强度 A 范围（越靠近 1 越偏白越亮）",
        min_value=0.4,
        max_value=1.0,
        value=(0.7, 1.0),
        step=0.05,
    )

    uploaded = st.file_uploader("请选择一张图片（JPG/PNG）", type=["jpg", "jpeg", "png"])

    col_btn, col_download = st.columns([1, 3])
    with col_btn:
        run = st.button("生成随机雾图", type="primary")

    if uploaded is None:
        st.info("请先上传一张图片。")
        return

    image = Image.open(uploaded).convert("RGB")
    img_rgb = np.array(image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if not run:
        st.info("点击左侧按钮“生成随机雾图”来查看效果。")
        col1, _ = st.columns(2)
        with col1:
            st.subheader("原始图像")
            st.image(img_rgb, width="stretch")
        return

    with st.spinner("正在生成雾图，请稍候..."):
        hazy_bgr, info = add_random_haze(
            img_bgr,
            beta_range=(beta_min, beta_max),
            A_range=(A_min, A_max),
        )
    hazy_rgb = cv2.cvtColor(hazy_bgr, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("原始图像")
        st.image(img_rgb, width="stretch")
    with col2:
        st.subheader("加雾后图像（随机雾气）")
        st.image(hazy_rgb, width="stretch")

    st.markdown("### 本次随机雾气参数")
    st.table([{"参数": k, "数值": v} for k, v in info.items()])

    with col_download:
        buf = BytesIO()
        Image.fromarray(hazy_rgb).save(buf, format="PNG")
        buf.seek(0)
        st.download_button(
            "下载当前雾图（PNG）",
            data=buf,
            file_name="hazy_random.png",
            mime="image/png",
        )


if __name__ == "__main__":
    main()

