import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import shap
import joblib
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, roc_auc_score

st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="💳",
    layout="wide"
)

# Load saved artifacts
# Save these from your notebook first:
#   joblib.dump(best_cat,'best_cat.pkl')
#   joblib.dump(X_train,'X_train.pkl')
#   joblib.dump(X_test,'X_test.pkl')
#   joblib.dump(y_test,'y_test.pkl')
#   submission.to_csv('submission.csv', index=False)

@st.cache_resource
def load_artifacts():
    model = joblib.load('best_cat.pkl')
    X_train = joblib.load('X_train.pkl')
    X_test = joblib.load('X_test.pkl')
    y_test = joblib.load('y_test.pkl')
    sub = pd.read_csv('submission.csv')
    return model, X_train, X_test, y_test, sub

model, X_train, X_test, y_test, submission = load_artifacts()

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

#Sidebar navigation
st.sidebar.title("Credit Risk")
page = st.sidebar.radio("", [
    "Overview",
    "Model Performance",
    "Customer Segments",
    "SHAP Explainability",
    "Predictions"
])


# PAGE 1: OVERVIEW

if page == "Overview":
    st.title("Credit Risk Default Prediction")
    st.caption("CatBoost - best model ")

    # KPI row
    auc = roc_auc_score(y_test, y_prob)
    from scipy.stats import ks_2samp
    ks = ks_2samp(y_prob[y_test == 1], y_prob[y_test == 0]).statistic
    dr = y_test.mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", f"{auc:.3f}")
    c2.metric("KS Statistic", f"{ks:.3f}")
    c3.metric("Default Rate", f"{dr*100:.1f}%")
    c4.metric("Test Samples", f"{len(y_test):,}")

    st.divider()

    col1, col2 = st.columns([1.4, 1])

    with col1:
        st.subheader("Model comparison")
        _m = pd.read_csv('metrics.csv').set_index('Model')
        model_scores = _m[['AUC-ROC', 'F1-Score', 'KS Statistic']]
        model_scores.columns = ['ROC-AUC', 'F1', 'KS']
        st.dataframe(
            model_scores.style.highlight_max(axis=0, color='#dbeafe'),
            use_container_width=True
        )

    with col2:
        st.subheader("Risk band distribution")
        submission['risk_band'] = pd.cut(
            submission['default_12m'],
            bins=[0, 0.1, 0.3, 0.6, 1.0],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        band_counts = submission['risk_band'].value_counts().sort_index()
        colors = ['#639922', '#EF9F27', '#D85A30', '#E24B4A']
        fig, ax = plt.subplots(figsize=(4, 3))
        bars = ax.bar(band_counts.index, band_counts.values, color=colors, width=0.6)
        ax.bar_label(bars, fmt='%d', padding=3, fontsize=9)
        ax.set_ylabel("Count")
        ax.set_title("Predicted risk bands (test set)", fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.divider()
    st.subheader("Top SHAP features (CatBoost)")

    @st.cache_data
    def compute_shap(_model, _X):
        explainer = shap.TreeExplainer(_model)
        shap_vals = explainer.shap_values(_X)
        mean_abs = pd.Series(
            np.abs(shap_vals).mean(axis=0),
            index=_X.columns
        ).sort_values(ascending=False).head(10)
        return mean_abs

    top_shap = compute_shap(model, X_test)
    fig, ax  = plt.subplots(figsize=(8, 3))
    ax.barh(top_shap.index[::-1], top_shap.values[::-1], color='#378ADD')
    ax.set_xlabel("Mean |SHAP value|")
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', alpha=0.3)
    st.pyplot(fig, use_container_width=True)
    plt.close()

# PAGE 2: MODEL PERFORMANCE
elif page == "Model Performance":
    st.title("Model Performance")

    tab1, tab2, tab3 = st.tabs(["ROC curve", "Precision-Recall", "Confusion matrix"])

    with tab1:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color='#185FA5', lw=2, label=f'CatBoost (AUC = {auc:.3f})')
        ax.plot([0,1],[0,1], 'k--', lw=0.8, alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        ax.spines[['top', 'right']].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with tab2:
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(rec, prec, color='#D85A30', lw=2)
        ax.axhline(y_test.mean(), color='gray', linestyle='--',
                   lw=0.8, label=f'Baseline ({y_test.mean():.3f})')
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend()
        ax.spines[['top', 'right']].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with tab3:
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(['Non-default', 'Default'])
        ax.set_yticklabels(['Non-default', 'Default'])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                        fontsize=13, color='white' if cm[i,j] > cm.max()/2 else 'black')
        plt.colorbar(im, ax=ax)
        st.pyplot(fig, use_container_width=True)
        plt.close()

# PAGE 3: CUSTOMER SEGMENTS
elif page == "Customer Segments":
    st.title("Performance Across Customer Segments")

    segment_df = X_test.copy()
    segment_df['actual_default'] = y_test.values
    segment_df['predicted_prob'] = y_prob

    segment_df['bureau_band'] = pd.cut(
        segment_df['bureau_score'],
        bins=[300, 550, 650, 750, 900],
        labels=['Poor', 'Fair', 'Good', 'Excellent']
    )
    segment_df['utilization_band'] = pd.cut(
        segment_df['utilization_ratio'],
        bins=[-0.01, 0.3, 0.7, 1.01],
        labels=['Low', 'Medium', 'High']
    )
    segment_df['delinquency_band'] = pd.cut(
        segment_df['delinquency_rate'],
        bins=[-0.01, 0, 0.25, 1],
        labels=['None', 'Low', 'High']
    )

    chosen = st.selectbox("Segment by", [
        'bureau_band', 'utilization_band', 'delinquency_band'
    ])

    analysis = segment_df.groupby(chosen, observed=True).apply(
        lambda g: pd.Series({
            'Count':            len(g),
            'Actual default %': round(g['actual_default'].mean() * 100, 2),
            'Predicted prob %': round(g['predicted_prob'].mean() * 100, 2),
            'ROC-AUC':          round(roc_auc_score(
                                    g['actual_default'], g['predicted_prob']
                                ), 3) if g['actual_default'].nunique() > 1 else float('nan')
        })
    ).reset_index()

    st.dataframe(analysis, use_container_width=True)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = range(len(analysis))
    w = 0.35
    ax.bar([i - w/2 for i in x], analysis['Actual default %'],
           w, label='Actual default %', color='#E24B4A', alpha=0.85)
    ax.bar([i + w/2 for i in x], analysis['Predicted prob %'],
           w, label='Predicted prob %', color='#378ADD', alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(analysis[chosen], rotation=15, ha='right')
    ax.set_ylabel("Rate (%)")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    st.pyplot(fig, use_container_width=True)
    plt.close()

# PAGE 4: SHAP EXPLAINABILITY
elif page == "SHAP Explainability":
    st.title("SHAP Explainability")
    st.caption("Based on CatBoost (best model)")

    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_test)

    tab1, tab2 = st.tabs(["Summary plot", "Single prediction"])

    with tab1:
        fig, ax = plt.subplots(figsize=(7, 5))
        shap.summary_plot(shap_vals, X_test, plot_type='bar',
                          max_display=15, show=False)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with tab2:
        idx = st.slider("Select test sample index", 0, len(X_test) - 1, 0)
        st.write(f"Predicted probability: **{y_prob[idx]:.3f}** | "
                 f"Actual: **{'Default' if y_test.iloc[idx] == 1 else 'Non-default'}**")
        fig, ax = plt.subplots(figsize=(7, 4))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_vals[idx],
                base_values=explainer.expected_value,
                data=X_test.iloc[idx],
                feature_names=X_test.columns.tolist()
            ),
            show=False
        )
        st.pyplot(fig, use_container_width=True)
        plt.close()

# PAGE 5: PREDICTIONS
elif page == "Predictions":
    st.title("Test Set Predictions")

    threshold = st.slider("Classification threshold", 0.1, 0.9, 0.5, 0.05)

    sub_display = submission.copy()
    sub_display['risk_band'] = pd.cut(
        sub_display['default_12m'],
        bins=[0, 0.1, 0.3, 0.6, 1.0],
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    sub_display['flag'] = (sub_display['default_12m'] >= threshold).map(
        {True: 'Default', False: 'Non-default'}
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Total predictions",   f"{len(sub_display):,}")
    c2.metric("Flagged as default",   f"{(sub_display['flag']=='Default').sum():,}")
    c3.metric("Flag rate",
              f"{(sub_display['flag']=='Default').mean()*100:.1f}%")

    st.dataframe(
        sub_display[['id', 'default_12m', 'risk_band', 'flag']]
        .sort_values('default_12m', ascending=False)
        .reset_index(drop=True),
        use_container_width=True
    )

    st.download_button(
        label="Download submission.csv",
        data=submission[['id', 'default_12m']].to_csv(index=False),
        file_name="submission.csv",
        mime="text/csv"
    )