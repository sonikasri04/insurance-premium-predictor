import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(page_title="Insurance Premium Predictor", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    * { font-family: 'Times New Roman', Times, serif !important; }
    .block-container { padding: 2rem 3rem; }
    .stButton>button {
        background-color: #2563eb; color: white; border: none;
        padding: 0.6rem 2rem; border-radius: 8px; font-size: 16px;
        font-weight: 600; width: 100%; margin-top: 1rem;
    }
    .stButton>button:hover { background-color: #1d4ed8; }
    .metric-card {
        background: white; border-radius: 12px; padding: 1.2rem 1.5rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08); text-align: center;
    }
    .metric-label { font-size: 13px; color: #64748b; margin-bottom: 4px; }
    .metric-value { font-size: 26px; font-weight: 700; color: #1e293b; }
    .result-box {
        background: #eff6ff; border: 1.5px solid #2563eb;
        border-radius: 12px; padding: 1.5rem; text-align: center; margin-top: 1rem;
    }
    .result-label { font-size: 14px; color: #2563eb; font-weight: 600; }
    .result-amount { font-size: 42px; font-weight: 800; color: #1e40af; }
    .section-title { font-size: 13px; font-weight: 600; color: #94a3b8;
        text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.8rem; }
    div[data-testid="stSlider"] > div { padding-top: 0; }
    .stSelectbox label, .stSlider label { font-size: 14px; color: #374151; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

model = joblib.load('insurance_model.pkl')

# ── Header ───────────────────────────────────────────────
st.markdown("## 🏥 Insurance Premium Predictor")
st.markdown("<p style='color:#64748b;margin-top:-12px'>Estimate your annual health insurance cost</p>", unsafe_allow_html=True)
st.divider()

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("<div class='section-title'>Personal Details</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age", 18, 64, 30)
        bmi = st.slider("BMI", 15.0, 55.0, 30.0, step=0.1)
    with c2:
        children = st.slider("Children", 0, 5, 0)
        sex = st.selectbox("Sex", ["Female", "Male"])

    st.markdown("<div class='section-title' style='margin-top:1rem'>Health & Location</div>", unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        smoker = st.selectbox("Smoker?", ["No", "Yes"])
    with c4:
        region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

    predict_btn = st.button("Calculate Premium")

with right:
    if predict_btn:
        sex_val = 1 if sex == "Male" else 0
        smoker_val = 1 if smoker == "Yes" else 0
        region_nw = 1 if region == "Northwest" else 0
        region_se = 1 if region == "Southeast" else 0
        region_sw = 1 if region == "Southwest" else 0
        bmi_smoker = bmi * smoker_val
        age_smoker = age * smoker_val
        age_bmi = age * bmi

        input_data = np.array([[age, sex_val, bmi, children, smoker_val,
                                region_nw, region_se, region_sw,
                                bmi_smoker, age_smoker, age_bmi]])
        prediction = model.predict(input_data)[0]
        monthly = prediction / 12

        st.markdown(f"""
        <div class='result-box'>
            <div class='result-label'>ESTIMATED ANNUAL PREMIUM</div>
            <div class='result-amount'>${prediction:,.0f}</div>
            <div style='color:#3b82f6;font-size:15px;margin-top:4px'>≈ ${monthly:,.0f} / month</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Risk factor bars ──────────────────────────────
        # st.markdown("<div class='section-title'>Risk Factor Breakdown</div>", unsafe_allow_html=True)

        # factors = {
        #     "Smoking": smoker_val * 100,
        #     "BMI": min((bmi - 15) / 40 * 100, 100),
        #     "Age": (age - 18) / 46 * 100,
        #     "Children": children / 5 * 100,
        # }

        # fig, ax = plt.subplots(figsize=(5, 2.4))
        # fig.patch.set_facecolor('#f8fafc')
        # ax.set_facecolor('#f8fafc')
        # colors = ['#ef4444' if v > 66 else '#f59e0b' if v > 33 else '#22c55e' for v in factors.values()]

        # bars = ax.barh(list(factors.keys()), list(factors.values()),
        #                color=colors, height=0.5, edgecolor='none')
        # ax.set_xlim(0, 100)
        # ax.set_xlabel("Risk Level (%)", fontsize=10, color='#64748b')
        # ax.tick_params(colors='#374151', labelsize=10)
        # ax.spines[['top','right','bottom','left']].set_visible(False)
        # ax.xaxis.set_tick_params(length=0)
        # ax.yaxis.set_tick_params(length=0)

        # for bar, val in zip(bars, factors.values()):
        #     ax.text(val + 1.5, bar.get_y() + bar.get_height()/2,
        #             f'{val:.0f}%', va='center', fontsize=9, color='#374151')

        # legend = [mpatches.Patch(color='#22c55e', label='Low'),
        #           mpatches.Patch(color='#f59e0b', label='Medium'),
        #           mpatches.Patch(color='#ef4444', label='High')]
        # ax.legend(handles=legend, loc='lower right', fontsize=8,
        #           frameon=False, labelcolor='#64748b')
        # plt.tight_layout()
        # st.pyplot(fig)

        # ── Summary metrics ───────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Summary</div>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        risk_label = "🔴 High" if prediction > 20000 else "🟡 Medium" if prediction > 8000 else "🟢 Low"
        with m1:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Risk Level</div><div class='metric-value' style='font-size:18px'>{risk_label}</div></div>", unsafe_allow_html=True)
        with m2:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Monthly Cost</div><div class='metric-value'>${monthly:,.0f}</div></div>", unsafe_allow_html=True)
        with m3:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>BMI Category</div><div class='metric-value' style='font-size:18px'>{'Obese' if bmi>=30 else 'Overweight' if bmi>=25 else 'Normal'}</div></div>", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style='background:white;border-radius:12px;padding:3rem 2rem;text-align:center;
        box-shadow:0 1px 4px rgba(0,0,0,0.08);margin-top:1rem'>
            <div style='font-size:48px'>🏥</div>
            <div style='font-size:18px;font-weight:600;color:#1e293b;margin:12px 0 6px'>Ready to Calculate</div>
            <div style='color:#64748b;font-size:14px'>Fill in your details on the left<br>and click Calculate Premium</div>
        </div>
        """, unsafe_allow_html=True)

# st.divider()
# st.markdown("<p style='text-align:center;color:#94a3b8;font-size:12px'>Built with Random Forest ML · R² = 0.875 · Training data: 1,338 records</p>", unsafe_allow_html=True)