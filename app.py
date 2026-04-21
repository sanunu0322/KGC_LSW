import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# --- [수정 1] 한글 폰트 설정 (리눅스 서버 환경 대응) ---
# 스트림릿 클라우드(리눅스) 환경에서는 기본 폰트에 한글이 없어 깨질 수 있습니다.
# 여기서는 차트 내 한글을 피하거나, 영문으로 표기하여 에러를 방지합니다.

# --- 모델 및 데이터 로드 (캐싱 적용) ---
@st.cache_resource
def load_resources():
    try:
        # [체크] 파일명이 깃허브에 올린 것과 토씨 하나 안 틀리고 똑같아야 합니다. (대소문자 구분)
        ml_data = joblib.load('kgc_model.pkl')
        dl_meta = joblib.load('dl_metadata.pkl')
        
        # [수정 2] 파일 확장자 확인 필수: .h5인지 .keras인지 확인 후 수정하세요.
        # 만약 깃허브에 kgc_lstm_model.h5가 있다면 아래 줄을 .h5로 고쳐야 합니다.
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
    st.warning("⚠️ 모델 파일(.pkl, .h5, .keras)이 GitHub 저장소의 루트 폴더에 있는지 확인해주세요.")
    st.info("💡 팁: 파일명이 대소문자까지 일치하는지, 확장자가 올바른지 확인이 필요합니다.")
    st.stop()

# 모델 및 스케일러 할당
clf_model = ml_data['model']
scaler = dl_meta['scaler']
# [수정 3] 학습 시 사용된 피처 순서 보장 (에러 방지)
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
# 컬럼 순서 맞춤
ml_input = ml_input[feature_names]
ml_prob = clf_model.predict_proba(ml_input)[0][1]

# [DL 예측 - RUL]
# [수정 4] 입력 데이터 구성 시 컬럼 순서 엄격 준수
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
    st.metric("예상 잔여 수명", f"{max(0, pred_rul):.1f} Cycles") # 음수 방지

    # [수정 5] 시각화 개선 (한글 깨짐 방지를 위해 영문 라벨 사용)
    fig, ax = plt.subplots(figsize=(6, 2))
    max_life = 200 # 가상의 최대 수명
    ax.barh(["RUL"], [max_life], color='#f0f2f6') # 배경
    
    # 수명에 따른 색상 변경
    bar_color = '#2ecc71' if pred_rul > 100 else '#f1c40f' if pred_rul > 50 else '#e74c3c'
    ax.barh(["RUL"], [min(max_life, pred_rul)], color=bar_color)
    
    ax.set_xlim(0, max_life)
    ax.set_title("Remaining Useful Life Gauge")
    st.pyplot(fig)

st.divider()
st.info(f"💡 **AI 분석 의견:** 현재 {temp}도의 온도 추세가 지속될 경우, 설비는 약 {int(max(0, pred_rul))}사이클 후에 점검이 필요할 것으로 예측됩니다.")
