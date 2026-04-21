import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# --- 모델 및 데이터 로드 (캐싱 적용) ---
@st.cache_resource
def load_resources():
    try:
        # 1. ML 모델 로드 (파일명: kgc_model.pkl)
        ml_data = joblib.load('kgc_model.pkl')
        
        # 2. DL 메타데이터 로드 (파일명: dl_metadata.pkl)
        dl_meta = joblib.load('dl_metadata.pkl')
        
        # 3. LSTM 모델 로드 
        # [수정] .h5 대신 호환성이 더 좋은 .keras 파일을 로드하도록 설정했습니다.
        # 깃허브 사진에 kgc_lstm_model.keras가 있는 것을 확인했습니다.
        lstm_model = load_model('kgc_lstm_model.keras') 
        
        return ml_data, dl_meta, lstm_model
    except Exception as e:
        st.error(f"❌ 파일 로드 중 오류 발생: {e}")
        return None, None, None

# 리소스 불러오기
ml_data, dl_meta, lstm_model = load_resources()

# 페이지 설정
st.set_page_config(page_title="KGC AI 통합 진단 시스템", layout="wide")
st.title("🛡️ KGC 부여공장 AI 예지보전 통합 대시보드")

# 모델 로드 실패 시 중단
if ml_data is None or dl_meta is None or lstm_model is None:
    st.warning("⚠️ GitHub 저장소에서 모델 파일을 읽어오지 못했습니다.")
    st.info("💡 팁: requirements.txt의 tensorflow 버전과 모델 저장 버전이 일치해야 합니다.")
    st.stop()

# 모델 및 스케일러 할당
clf_model = ml_data['model']
scaler = dl_meta['scaler']
feature_names = dl_meta.get('features', ['증삼기_내부온도', '추출기_상단온도', '건조기_출구온도', '이송펌프_회전속도', '가열히터_전류값'])

# --- 사이드바: 실시간 데이터 입력 ---
st.sidebar.header("📡 실시간 센서 데이터")
temp = st.sidebar.slider("🌡️ 증삼기 내부온도", 630.0, 650.0, 642.0)
press = st.sidebar.slider("🔥 건조기 출구온도", 1390.0, 1440.0, 1415.0)
speed = st.sidebar.slider("⚙️ 이송펌프 회전속도", 500, 2500, 1200)

# --- 예측 로직 ---
# [ML 예측]
ml_input = pd.DataFrame([{
    '증삼기_내부온도': temp, 
    '추출기_상단온도': 1598, 
    '건조기_출구온도': press, 
    '이송펌프_회전속도': speed, 
    '가열히터_전류값': 520
}])
ml_input = ml_input[feature_names]
ml_prob = clf_model.predict_proba(ml_input)[0][1]

# [DL 예측 - RUL]
current_val = [temp, 1598, press, speed, 520]
base_seq = np.tile(current_val, (50, 1))
scaled_seq = scaler.transform(base_seq)
dl_input = np.expand_dims(scaled_seq, axis=0)
pred_rul = lstm_model.predict(dl_input, verbose=0)[0][0]

# --- 화면 구성 ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🤖 머신러닝: 실시간 상태 진단")
    st.metric("이상 징후 확률", f"{ml_prob*100:.1f}%")
    if ml_prob > 0.5:
        st.error("🚨 진단 결과: 설비 이상 감지")
    else:
        st.success("✅ 진단 결과: 정상 가동 중")

with col2:
    st.subheader("🧠 딥러닝: 잔여 수명 예측 (RUL)")
    display_rul = max(0, pred_rul)
    st.metric("예상 잔여 수명", f"{display_rul:.1f} Cycles")

    # 시각화 (영문 라벨 사용으로 깨짐 방지)
    fig, ax = plt.subplots(figsize=(6, 2))
    max_life = 200
    ax.barh(["RUL"], [max_life], color='#f0f2f6')
    
    bar_color = '#2ecc71' if display_rul > 100 else '#f1c40f' if display_rul > 50 else '#e74c3c'
    ax.barh(["RUL"], [min(max_life, display_rul)], color=bar_color)
    
    ax.set_xlim(0, max_life)
    ax.set_title("Remaining Useful Life Gauge")
    st.pyplot(fig)

st.divider()
st.info(f"💡 **AI 분석 의견:** 현재 {temp}도의 온도 추세가 지속될 경우, 설비는 약 {int(display_rul)}사이클 후에 점검이 필요할 것으로 예측됩니다.")
