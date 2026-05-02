import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

# ── Load data ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("churn_predictions.csv")
    return df

df = load_data()

# ── Header ────────────────────────────────────────────────
st.title("📊 Customer Churn Intelligence Dashboard")
st.markdown("*AI-Augmented Churn Prediction · XGBoost + SHAP · Telco Dataset*")
st.divider()

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.header("🔍 Filters")
segment_filter = st.sidebar.multiselect(
    "Customer Segment",
    options=df['Segment'].unique(),
    default=df['Segment'].unique()
)
threshold = st.sidebar.slider(
    "Churn Risk Threshold",
    min_value=0.1, max_value=0.9,
    value=0.5, step=0.05
)

filtered_df = df[df['Segment'].isin(segment_filter)].copy()
filtered_df['Risk_Label'] = filtered_df['Churn_Probability'].apply(
    lambda x: 'High Risk' if x >= threshold else 'Low Risk'
)

# ── KPI Metrics ───────────────────────────────────────────
st.subheader("📈 Key Metrics")
col1, col2, col3, col4 = st.columns(4)

total = len(filtered_df)
high_risk = len(filtered_df[filtered_df['Churn_Probability'] >= threshold])
avg_prob = filtered_df['Churn_Probability'].mean()
actual_churn_rate = filtered_df['Actual_Churn'].mean()

col1.metric("Total Customers", f"{total:,}")
col2.metric("High Risk Customers", f"{high_risk:,}",
            delta=f"{high_risk/total*100:.1f}% of total",
            delta_color="inverse")
col3.metric("Avg Churn Probability", f"{avg_prob:.1%}")
col4.metric("Actual Churn Rate", f"{actual_churn_rate:.1%}")

st.divider()

# ── Charts Row 1 ──────────────────────────────────────────
st.subheader("🔎 Churn Risk Analysis")
col1, col2 = st.columns(2)

with col1:
    # Churn probability distribution
    fig1 = px.histogram(
        filtered_df, x='Churn_Probability',
        color='Segment',
        nbins=30,
        title='Churn Probability Distribution by Segment',
        color_discrete_map={
            'At-Risk': '#E24B4A',
            'Champions': '#1D9E75',
            'Budget Loyalists': '#378ADD'
        }
    )
    fig1.add_vline(x=threshold, line_dash="dash",
                   line_color="black",
                   annotation_text=f"Threshold: {threshold}")
    fig1.update_layout(bargap=0.1)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    # Churn rate by segment
    seg_stats = filtered_df.groupby('Segment').agg(
        Churn_Rate=('Actual_Churn', 'mean'),
        Avg_Risk=('Churn_Probability', 'mean'),
        Count=('Actual_Churn', 'count')
    ).reset_index()

    fig2 = px.bar(
        seg_stats, x='Segment', y='Churn_Rate',
        color='Segment',
        title='Actual Churn Rate by Segment',
        text=seg_stats['Churn_Rate'].apply(lambda x: f'{x:.1%}'),
        color_discrete_map={
            'At-Risk': '#E24B4A',
            'Champions': '#1D9E75',
            'Budget Loyalists': '#378ADD'
        }
    )
    fig2.update_traces(textposition='outside')
    fig2.update_layout(showlegend=False, yaxis_tickformat='.0%')
    st.plotly_chart(fig2, use_container_width=True)

# ── Charts Row 2 ──────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    # Tenure vs Churn Probability scatter
    fig3 = px.scatter(
        filtered_df, x='tenure', y='Churn_Probability',
        color='Segment',
        title='Tenure vs Churn Probability',
        opacity=0.5,
        color_discrete_map={
            'At-Risk': '#E24B4A',
            'Champions': '#1D9E75',
            'Budget Loyalists': '#378ADD'
        }
    )
    fig3.add_hline(y=threshold, line_dash="dash",
                   line_color="black",
                   annotation_text="Risk threshold")
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    # Monthly charges vs churn probability
    fig4 = px.scatter(
        filtered_df, x='MonthlyCharges', y='Churn_Probability',
        color='Segment',
        title='Monthly Charges vs Churn Probability',
        opacity=0.5,
        color_discrete_map={
            'At-Risk': '#E24B4A',
            'Champions': '#1D9E75',
            'Budget Loyalists': '#378ADD'
        }
    )
    fig4.add_hline(y=threshold, line_dash="dash",
                   line_color="black")
    st.plotly_chart(fig4, use_container_width=True)

st.divider()

# ── High Risk Customer Table ───────────────────────────────
st.subheader("🚨 High Risk Customers")
high_risk_df = filtered_df[filtered_df['Churn_Probability'] >= threshold]\
    .sort_values('Churn_Probability', ascending=False)\
    [['Segment', 'tenure', 'MonthlyCharges',
      'Contract', 'Churn_Probability', 'Actual_Churn']]\
    .head(20)

high_risk_df['Churn_Probability'] = high_risk_df['Churn_Probability']\
    .apply(lambda x: f"{x:.1%}")
high_risk_df['Contract'] = high_risk_df['Contract'].map(
    {0: 'Month-to-month', 1: 'One year', 2: 'Two year'})
high_risk_df['Actual_Churn'] = high_risk_df['Actual_Churn']\
    .map({1: '✅ Churned', 0: '❌ Stayed'})

st.dataframe(high_risk_df, use_container_width=True)

st.divider()

# ── AI Retention Memo Generator ───────────────────────────
st.subheader("🤖 AI Retention Memo Generator")
st.markdown("Select a customer segment to generate a plain-English retention strategy.")

selected_segment = st.selectbox(
    "Choose a segment:",
    options=['At-Risk', 'Champions', 'Budget Loyalists']
)

if st.button("⚡ Generate Retention Memo", type="primary"):
    seg_data = filtered_df[filtered_df['Segment'] == selected_segment]
    avg_tenure = seg_data['tenure'].mean()
    avg_charges = seg_data['MonthlyCharges'].mean()
    churn_rate = seg_data['Actual_Churn'].mean()
    size = len(seg_data)

    # ── Mock AI memo (swap this for real API call later) ──
    memos = {
        'At-Risk': f"""
**RETENTION MEMO — AT-RISK SEGMENT**
*Generated by AI Analysis Engine*

**Segment Overview**
{size:,} customers · Avg tenure {avg_tenure:.0f} months · 
Avg monthly charges ${avg_charges:.2f} · Churn rate {churn_rate:.1%}

**Key Finding**
This segment represents your most urgent retention priority. 
Customers here are relatively new ({avg_tenure:.0f} months avg) 
but paying premium prices (${avg_charges:.2f}/month), creating 
a high-risk value mismatch. Nearly half will churn without intervention.

**Recommended Actions**
1. **Immediate outreach** — Contact all At-Risk customers within 7 days 
   via personalized email citing their specific usage patterns.
2. **Contract upgrade incentive** — Offer 15% discount to switch from 
   month-to-month to annual contract. SHAP analysis shows contract type 
   is the #1 churn driver.
3. **Add-on bundle** — Offer free 3-month trial of OnlineSecurity + 
   TechSupport. These features reduce churn probability significantly 
   per model explainability analysis.
4. **Fiber optic review** — Audit pricing for fiber optic customers 
   specifically. This group shows anomalous churn suggesting a 
   price-to-value perception problem.

**Expected Impact**
If 30% of At-Risk customers accept the contract upgrade offer, 
estimated annual revenue retention: ${size * avg_charges * 0.30 * 12:,.0f}
        """,
        'Champions': f"""
**RETENTION MEMO — CHAMPIONS SEGMENT**
*Generated by AI Analysis Engine*

**Segment Overview**
{size:,} customers · Avg tenure {avg_tenure:.0f} months · 
Avg monthly charges ${avg_charges:.2f} · Churn rate {churn_rate:.1%}

**Key Finding**
Champions are your most valuable and stable customers. 
With {avg_tenure:.0f} months average tenure and only {churn_rate:.1%} 
churn rate, this segment drives disproportionate revenue. 
Protect and expand this base.

**Recommended Actions**
1. **Loyalty rewards** — Launch a VIP program for customers 
   with tenure > 48 months. Recognition reduces churn even 
   among satisfied customers.
2. **Upsell opportunity** — Champions already pay ${avg_charges:.2f}/month. 
   Premium tier upgrades (higher bandwidth, priority support) 
   have high acceptance rates in this cohort.
3. **Referral program** — Champions are your best acquisition channel. 
   A referral incentive converts brand loyalty into growth.
4. **Proactive support** — Assign dedicated account managers to 
   top 10% by revenue. Prevents silent dissatisfaction.

**Expected Impact**
Reducing Champions churn by just 2 percentage points retains 
approximately {int(size * 0.02):,} high-value customers worth 
${int(size * 0.02 * avg_charges * 12):,} in annual revenue.
        """,
        'Budget Loyalists': f"""
**RETENTION MEMO — BUDGET LOYALISTS SEGMENT**
*Generated by AI Analysis Engine*

**Segment Overview**
{size:,} customers · Avg tenure {avg_tenure:.0f} months · 
Avg monthly charges ${avg_charges:.2f} · Churn rate {churn_rate:.1%}

**Key Finding**
Budget Loyalists are stable low-cost customers with surprisingly 
good retention ({churn_rate:.1%} churn). At ${avg_charges:.2f}/month 
they are below average revenue contributors but represent a 
large, reliable base with upgrade potential.

**Recommended Actions**
1. **Upgrade campaign** — Target customers with tenure > 24 months 
   with mid-tier plan offers. Long tenure signals satisfaction; 
   they are most likely to accept upgrades.
2. **Price anchoring** — Show Budget Loyalists the value gap between 
   their current plan and mid-tier. Framing matters more than discount.
3. **Service expansion** — Introduce streaming bundles at marginal 
   cost increase. Low-price customers respond well to perceived value.
4. **Maintain simplicity** — Avoid over-engineering retention for 
   this segment. Their churn rate is already low; focus resources 
   on At-Risk instead.

**Expected Impact**
Converting 15% of Budget Loyalists to mid-tier plans increases 
segment revenue by ${int(size * 0.15 * 30 * 12):,} annually 
with minimal churn risk.
        """
    }

    with st.spinner("Generating retention strategy..."):
        import time
        time.sleep(1.5)  # simulate API call

    st.success("Memo generated!")
    st.markdown(memos[selected_segment])

    st.download_button(
        label="📄 Download Memo as TXT",
        data=memos[selected_segment],
        file_name=f"retention_memo_{selected_segment.lower().replace(' ', '_')}.txt",
        mime="text/plain"
    )

# ── Footer ────────────────────────────────────────────────
st.divider()
st.markdown(
    "*Built with XGBoost · SHAP · Streamlit · "
    "IBM Telco Churn Dataset · "
    "Model AUC-ROC: 0.837*"
)