from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predict import ( 
    combine_input,
    evaluate_model,
    get_top_k_predictions,
    load_and_prepare_data,
    train_model,
)


st.set_page_config(
    page_title="Job Trend IT Demo",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Job Trend IT - Demo")
st.caption("Demo Mô hình Dự đoán Vai trò IT (TF-IDF & Linear SVM)")


def load_artifacts():
    df = load_and_prepare_data()
    model, X_train, y_train, X_test, y_test = train_model(df)
    metrics = evaluate_model(model, X_test, y_test)
    return df, model, metrics, len(X_train), len(X_test)


def render_metrics(metrics: dict[str, float], train_size: int, test_size: int, num_classes: int) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    c2.metric("Macro F1", f"{metrics['macro_f1']:.4f}")
    c3.metric("Weighted F1", f"{metrics['weighted_f1']:.4f}")
    c4.metric("Test samples", f"{test_size}")

    st.caption(f"Train: {train_size} | Test: {test_size} | Classes: {num_classes}")


def render_confusion_image() -> None:
    cm_path = ROOT_DIR / "results" / "confusion_matrix.png"
    if cm_path.exists():
        st.image(str(cm_path), caption="Confusion matrix chính thức", use_container_width=True)


def render_single_prediction(model) -> None:
    st.subheader("Dự đoán một mẫu")
    with st.form("single_predict_form"):
        title = st.text_input(
            "Job title",
            value="Senior Backend Developer",
            placeholder="Ví dụ: Senior Backend Developer",
        )
        description = st.text_area(
            "Job description",
            value="Build REST APIs with Java, Spring Boot, MySQL, Docker and Kubernetes.",
            height=180,
        )
        top_k = st.slider("Top-k", min_value=1, max_value=5, value=3)
        submitted = st.form_submit_button("Predict")

    if submitted:
        if not title.strip() or not description.strip():
            st.warning("Vui lòng nhập đủ title và description.")
            return

        text = combine_input(title, description)
        predicted = model.predict([text])[0]
        top_preds = get_top_k_predictions(model, [text], top_k=top_k)[0]

        st.success(f"Nhãn dự đoán: **{predicted}**")
        st.write("Top nhãn gần nhất:")
        st.dataframe(
            pd.DataFrame(top_preds, columns=["Label", "Score"]),
            use_container_width=True,
            hide_index=True,
        )

def main() -> None:
    df, model, metrics, train_size, test_size = load_artifacts()

    st.subheader("Kết quả chính thức")
    render_metrics(metrics, train_size, test_size, int(df["nhan"].nunique()))

    with st.expander("Xem confusion matrix"):
        render_confusion_image()

    render_single_prediction(model)


if __name__ == "__main__":
    main()
