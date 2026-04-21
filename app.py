import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# 1. 모든 모델 및 데이터 로드
ml_data = joblib.load('kgc_model.pkl') # 기존 ML 모델
clf_model = ml_data['model']

dl_meta = joblib.load('dl_metadata.pkl') # DL 스케일러
scaler = dl_meta['scaler']
features = dl_meta['features']

lstm_model = load_model('kgc_lstm_model.h5') # DL LSTM 모델

st.set_page_config(page_title="KGC AI 통합 진단 시스템", layout="wide")
st.title("🛡️ KGC 부여공장 AI 예지보전 통합 대시보드")

# 2. 사이드바: 실시간 데이터 입력
st.sidebar.header("📡 실시간 센서 데이터")
temp = st.sidebar.slider("증삼기 내부온도", 630.0, 650.0, 642.0)
press = st.sidebar.slider("건조기 출구온도", 1390.0, 1440.0, 1415.0)
speed = st.sidebar.slider("이송펌프 회전속도", 500, 2500, 1200)

# 3. 예측 로직
# [ML용 데이터]
ml_input = pd.DataFrame([{'증삼기_내부온도': temp, '추출기_상단온도': 1598, '건조기_출구온도': press, '이송펌프_회전속도': speed, '가열히터_전류값': 520}])
ml_prob = clf_model.predict_proba(ml_input)[0][1]

# [DL용 데이터] - LSTM은 50개의 시퀀스가 필요하므로, 현재 값을 기준으로 가상의 시퀀스 생성
# 실제 현장이라면 최근 50개의 로그를 DB에서 긁어와야 합니다.
base_seq = np.tile([temp, 1598, press, speed, 520], (50, 1)) # 현재 값으로 50개 채움
scaled_seq = scaler.transform(base_seq) # 학습 때와 동일하게 스케일링
dl_input = np.expand_dims(scaled_seq, axis=0)
pred_rul = lstm_model.predict(dl_input, verbose=0)[0][0]

# 4. 화면 구성
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
    ax.barh(["Remaining Life"], [pred_rul], color='gold' if pred_rul > 50 else 'orange')
    ax.set_xlim(0, 200) # 최대 수명을 200으로 가정
    st.pyplot(fig)

st.divider()
st.info(f"💡 **AI 분석 의견:** 현재 {temp}도의 온도 추세가 지속될 경우, 설비는 약 {int(pred_rul)}일 후에 임계치에 도달할 것으로 예측됩니다.")
