import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from cc_ocr_date_extract.pipelines.ml.nodes import evaluate_model
from cc_ocr_date_extract.pipelines.ml.utils import apply_threshold

st.title('Control - Scenarios')


# @st.cache()
def cached_predictions():
    with open('data/06_models/date_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    df_test = pd.read_csv(
        'data/05_model_input/test_formatted.csv')

    return evaluate_model(loaded_model, df_test)


res, df_docs = cached_predictions()

threshold = st.slider('Select decision threshold', min_value=0.00, max_value=1.0, step=0.01, value=0.8)

names = ['tn', 'fp', 'fn', 'tp', 'n_pred_pos', 'n_docs_manual', 'threshold', 'p']
metrics = pd.DataFrame({'name': names, 'value': apply_threshold(df_docs['label'], df_docs['predict_proba_belegdatum'],
                                                                pos_label='Belegdatum', threshold=threshold)})
st.text('Evaluation result')
st.dataframe(metrics)

st.text('Prediction probability')

fig, ax = plt.subplots()
fig.set_figheight(3)

df_docs['predict_proba_belegdatum'].hist(bins=30, rwidth=1, ax=ax)

ax.axvline(threshold, 0, 1, color='turquoise', linewidth=2)

# Add plot labels
ax.set(xlabel='Prediction probability', ylabel='Count')
ax.set(yticklabels=[])
ax.set_title('Visualization')

ax.set_xlim(0, 1)
plt.tight_layout()

st.pyplot(fig)
