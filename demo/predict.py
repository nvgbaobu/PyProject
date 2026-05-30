"""
demo/predict.py – Demo phân loại vai trò IT
Nhóm 17 – Job Trend IT – Tuần 4

Cách chạy:
    python demo/predict.py
    python demo/predict.py --input data/raw/job_descriptions.csv
"""
import argparse, os, re, sys
import pandas as pd

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import train_test_split
except ImportError:
    print("Lỗi: pip install scikit-learn pandas")
    sys.exit(1)

RANDOM_SEED = 42
NGUONG_GOM  = 50

BANNER = """
╔══════════════════════════════════════════════════════════╗
║     JOB TREND IT – Phân loại vai trò CNTT               ║
║     Nhóm 17 – Tuần 4 Demo                               ║
║     Nhập câu → dự đoán nhãn  (theo yêu cầu thầy mục 9) ║
╚══════════════════════════════════════════════════════════╝
"""

MAU_DEMO = [
    {'title': 'Backend Developer – Java Spring Boot',
     'description': 'Experienced backend developer needed. Build RESTful APIs using '
                    'Spring Boot, design MySQL schemas, integrate Redis caching. '
                    'Docker and CI/CD experience required.'},
    {'title': 'Frontend Developer ReactJS',
     'description': 'Build dynamic UI using ReactJS and TypeScript. Knowledge of '
                    'Redux, REST API integration, responsive design and Git required.'},
    {'title': 'QA Automation Engineer',
     'description': 'Build automation testing frameworks using Selenium and TestNG. '
                    'Write test scripts, regression testing, Jenkins integration.'},
    {'title': 'DevOps / Cloud Engineer',
     'description': 'Manage CI/CD pipelines using GitLab CI. Deploy infrastructure '
                    'on AWS with Terraform. Monitor with Prometheus/Grafana. '
                    'Docker and Kubernetes required.'},
    {'title': 'Data Scientist – Python ML',
     'description': 'Develop machine learning models using Python, scikit-learn, '
                    'TensorFlow. Experience with NLP, data analysis, SQL, Pandas.'},
]


def clean_text(text):
    text = str(text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^\w\s.+#/]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


def build_model(data_path):
    df = pd.read_csv(data_path)
    df['it_role_type'] = df['it_role_type'].replace(
        {'Fullstack Developer': 'Full-stack Developer'})
    df = df.drop_duplicates(subset=['title','company','description']).reset_index(drop=True)

    dem = df['it_role_type'].value_counts().to_dict()
    vai_tro_lon = [r for r,c in dem.items() if c >= NGUONG_GOM and r != 'Other IT Role']
    df['nhan']              = df['it_role_type'].apply(
        lambda x: x if x in vai_tro_lon else 'Other IT Role')
    df['description_clean'] = df['description'].apply(clean_text)
    df['title_sach']         = df['title'].apply(clean_text)
    df['X'] = df['title_sach'] + ' ' + df['title_sach'] + ' ' + df['description_clean']

    X_tr, _, y_tr, _ = train_test_split(
        df['X'], df['nhan'], test_size=0.2,
        random_state=RANDOM_SEED, stratify=df['nhan'])

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), sublinear_tf=True)
    clf   = LinearSVC(C=0.5, max_iter=3000, random_state=RANDOM_SEED, class_weight='balanced')
    clf.fit(tfidf.fit_transform(X_tr), y_tr)

    print(f"[✓] Mô hình sẵn sàng | {len(df)} mẫu | {df['nhan'].nunique()} lớp")
    return tfidf, clf


def predict_single(tfidf, clf, title, description):
    tc  = clean_text(title)
    dc  = clean_text(description)
    x   = tc + ' ' + tc + ' ' + dc
    vec = tfidf.transform([x])
    pred    = clf.predict(vec)[0]
    scores  = clf.decision_function(vec)[0]
    top3_idx = scores.argsort()[::-1][:3]
    top3     = [(clf.classes_[i], round(float(scores[i]),3)) for i in top3_idx]
    return pred, top3


def run_interactive(tfidf, clf):
    print(BANNER)
    print("Chọn số (1–5) để dùng mẫu sẵn | 'c' để tự nhập | 'q' để thoát\n")
    for i, s in enumerate(MAU_DEMO, 1):
        print(f"  [{i}] {s['title']}")
    print()

    while True:
        choice = input(">>> ").strip().lower()
        if choice == 'q':
            break
        elif choice == 'c':
            title = input("  Title       : ").strip()
            desc  = input("  Description : ").strip()
        elif choice.isdigit() and 1 <= int(choice) <= len(MAU_DEMO):
            s = MAU_DEMO[int(choice)-1]
            title, desc = s['title'], s['description']
            print(f"\n  Title : {title}")
            print(f"  Desc  : {desc[:90]}...")
        else:
            print("  Nhập 1–5, 'c' hoặc 'q'")
            continue

        pred, top3 = predict_single(tfidf, clf, title, desc)
        print(f"\n  ┌─ KẾT QUẢ ─────────────────────────────────")
        print(f"  │  Vai trò dự đoán : ► {pred}")
        print(f"  │  Top 3 ứng viên  :")
        for role, score in top3:
            print(f"  │    {role:<35} (score: {score:+.3f})")
        print(f"  └────────────────────────────────────────────\n")


def run_batch(tfidf, clf, input_path):
    df = pd.read_csv(input_path)
    if not {'title','description'}.issubset(df.columns):
        raise ValueError("File CSV cần có cột 'title' và 'description'")
    df['tc'] = df['title'].apply(clean_text)
    df['dc'] = df['description'].apply(clean_text)
    df['X']  = df['tc'] + ' ' + df['tc'] + ' ' + df['dc']
    df['predicted_role'] = clf.predict(tfidf.transform(df['X']))
    out = input_path.replace('.csv', '_predicted.csv')
    df[['title','predicted_role']].to_csv(out, index=False)
    print(df[['title','predicted_role']].to_string(index=False))
    print(f"\n✓ Đã lưu → {out}")


def main():
    parser = argparse.ArgumentParser()
    default_data_path = os.path.join(os.path.dirname(__file__), '../data/raw/job_descriptions.csv')
    parser.add_argument('--data',  default=default_data_path)
    parser.add_argument('--input', default=None)
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Không tìm thấy: {args.data}")
        print("Đặt file job_descriptions.csv vào data/raw/")
        sys.exit(1)

    print("[*] Đang huấn luyện...", end=' ', flush=True)
    tfidf, clf = build_model(args.data)

    if args.input:
        run_batch(tfidf, clf, args.input)
    else:
        run_interactive(tfidf, clf)


if __name__ == '__main__':
    main()
