from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data_sample" / "job_descriptions.csv"
RANDOM_SEED = 42
NGUONG_GOM = 50

ROLE_NORMALIZATION = {
    "Fullstack Developer": "Full-stack Developer",
    "Software Developer": "Software Engineer",
    "DevOps Engineer": "DevOps/Cloud/System Engineer",
    "Cloud Engineer": "DevOps/Cloud/System Engineer",
    "System Engineer": "DevOps/Cloud/System Engineer",
    "IT System Engineer": "DevOps/Cloud/System Engineer",
}

WHITELIST_ROLES = {
    "DevOps/Cloud/System Engineer",
    "Data Engineer",
    "Business Analyst",
}

REQUIRED_COLUMNS = {"title", "description", "it_role_type"}


def clean_text(text: object) -> str:
    text = str(text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^\w\s.+#/]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def load_and_prepare_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Khong tim thay file du lieu: {data_path}")

    df = pd.read_csv(data_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"File du lieu thieu cot bat buoc: {sorted(missing)}")

    if {"title", "company", "description"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["title", "company", "description"]).reset_index(drop=True)
    else:
        df = df.drop_duplicates(subset=["title", "description"]).reset_index(drop=True)

    df["it_role_type"] = df["it_role_type"].replace(ROLE_NORMALIZATION)

    label_counts = df["it_role_type"].value_counts()
    major_roles = [
        role
        for role in label_counts.index
        if (label_counts[role] >= NGUONG_GOM or role in WHITELIST_ROLES) and role != "Other IT Role"
    ]
    df["nhan"] = df["it_role_type"].apply(lambda x: x if x in major_roles else "Other IT Role")

    df["title_sach"] = df["title"].apply(clean_text)
    df["description_clean"] = df["description"].apply(clean_text)
    df["title_va_mo_ta"] = df["title_sach"] + " " + df["title_sach"] + " " + df["description_clean"]

    return df


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True),
            ),
            (
                "clf",
                LinearSVC(
                    C=0.5,
                    max_iter=3000,
                    random_state=RANDOM_SEED,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def train_model(df: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X = df["title_va_mo_ta"]
    y = df["nhan"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    return model, X_train, y_train, X_test, y_test


def combine_input(title: object, description: object) -> str:
    title_clean = clean_text(title)
    desc_clean = clean_text(description)
    return f"{title_clean} {title_clean} {desc_clean}"


def get_top_k_predictions(model: Pipeline, texts: Iterable[str], top_k: int = 3) -> list[list[tuple[str, float]]]:
    scores = model.decision_function(list(texts))
    scores = np.atleast_2d(scores)
    labels = model.classes_

    top_results: list[list[tuple[str, float]]] = []
    for row in scores:
        order = np.argsort(row)[::-1][:top_k]
        top_results.append([(labels[idx], float(row[idx])) for idx in order])
    return top_results


def evaluate_model(model: Pipeline, X_test: pd.Series, y_test: pd.Series) -> dict[str, float]:
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
    }


def predict_single(model: Pipeline, title: str, description: str, top_k: int = 3) -> None:
    text = combine_input(title, description)
    pred = model.predict([text])[0]
    top_preds = get_top_k_predictions(model, [text], top_k=top_k)[0]

    print("\n=== KET QUA DU DOAN ===")
    print(f"Nhan du doan: {pred}")
    print(f"Title      : {title}")
    print(f"Description: {description}")
    print("\nTop ung vien:")
    for rank, (label, score) in enumerate(top_preds, start=1):
        print(f"  {rank}. {label:<35} score={score:.4f}")


def predict_csv(model: Pipeline, input_path: Path, output_path: Path | None = None) -> pd.DataFrame:
    df_input = pd.read_csv(input_path)
    if not {"title", "description"}.issubset(df_input.columns):
        raise ValueError("File CSV dau vao phai co cot 'title' va 'description'.")

    combined = [
        combine_input(row["title"], row["description"])
        for _, row in df_input.iterrows()
    ]
    df_input["predicted_role"] = model.predict(combined)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_input.to_csv(output_path, index=False)

    return df_input


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and run the week 4 IT job role classifier.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_PATH,
        help="Duong dan file du lieu goc job_descriptions.csv",
    )
    parser.add_argument(
        "--title",
        type=str,
        help="Tieu de tin tuyen dung can du doan",
    )
    parser.add_argument(
        "--description",
        type=str,
        help="Mo ta tin tuyen dung can du doan",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="File CSV dau vao co cot title va description",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="File CSV dau ra neu dung --input",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="So nhan xep hang cao nhat muon in ra",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    df = load_and_prepare_data(args.data)
    model, X_train, y_train, X_test, y_test = train_model(df)

    metrics = evaluate_model(model, X_test, y_test)
    print("=== TONG QUAN MO HINH ===")
    print(f"Train: {len(X_train)} | Test: {len(X_test)} | So lop: {df['nhan'].nunique()}")
    print(
        "Accuracy={accuracy:.4f} | Macro F1={macro_f1:.4f} | Weighted F1={weighted_f1:.4f}".format(
            **metrics
        )
    )
    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, model.predict(X_test), digits=4))

    if args.input is not None:
        output_path = args.output if args.output is not None else None
        result_df = predict_csv(model, args.input, output_path=output_path)
        print(f"\nDa du doan xong {len(result_df)} dong tu file: {args.input}")
        if output_path is not None:
            print(f"Da luu ket qua tai: {output_path}")
        return

    if args.title and args.description:
        predict_single(model, args.title, args.description, top_k=args.top_k)
        return

    print("\nKhong co dau vao cu the, dung mot vi du mau de demo:")
    predict_single(
        model,
        "Senior Backend Developer",
        "Build REST APIs with Java, Spring Boot, MySQL, Docker and Kubernetes.",
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
