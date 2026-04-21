import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# --- 모델 및 데이터 로드 (캐싱 적용) ---
@st.cache_resource
def load_resources():
    try:
        # ML 모델 로드
        ml_data = joblib.load('kgc_model.pkl')
        # DL 메타데이터 로드
        dl_meta = joblib.load('dl_metadata.pkl')
        # LSTM 모델 로드
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

if ml_data is None or dl_meta is None or lstm_model is None:
    st.warning("⚠️ 모델 파일(.pkl, .h5)이 GitHub 저장소에 있는지 확인해주세요.")
    st.stop()

# 모델 변수 할당
clf_model = ml_data['model']
scaler = dl_meta['scaler']

# --- 사이드바: 실시간 데이터 입력 ---
st.sidebar.header("📡 실시간 센서 데이터")
temp = st.sidebar.slider("증삼기 내부온도", 630.0, 650.0, 642.0)
press = st.sidebar.slider("건조기 출구온도", 1390.0, 1440.0, 1415.0)
speed = st.sidebar.slider("이송펌프 회전속도", 500, 2500, 1200)

# --- 예측 로직 ---
# [ML 예측]
ml_input = pd.DataFrame([{
    '증삼기_내부온도': temp, 
    '추출기_상단온도': 1598, 
    '건조기_출구온도': press, 
    '이송펌프_회전속도': speed, 
    '가열히터_전류값': 520
}])
ml_prob = clf_model.predict_proba(ml_input)[0][1]

# [DL 예측 - RUL]
# 현재 입력값으로 50개 시퀀스 생성 (실제 데이터셋 구조에 맞춤)
base_seq = np.tile([temp, 1598, press, speed, 520], (50, 1))
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
    st.metric("예상 잔여 수명", f"{pred_rul:.1f} Cycles")

    # 수명 게이지 시각화
    fig, ax = plt.subplots(figsize=(5, 2))
    # 게이지 배경색 설정
    ax.barh(["Remaining Life"], [200], color='#eeeeee')
    # 실제 수명 표시
    color = 'gold' if pred_rul > 50 else 'orange'
    ax.barh(["Remaining Life"], [pred_rul], color=color)
    ax.set_xlim(0, 200)
    st.pyplot(fig)

st.divider()
st.info(f"💡 **AI 분석 의견:** 현재 {temp}도의 온도 추세가 지속될 경우, 설비는 약 {int(pred_rul)}사이클 후에 점검이 필요할 것으로 예측됩니다.")
