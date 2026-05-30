"""
demo/app.py – Streamlit Dashboard
Nhóm 17 – Job Trend IT – Tuần 4
Cách chạy: streamlit run demo/app.py
"""
# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
import numpy as np
import re
# pyrefly: ignore [missing-import]
import plotly.express as px
# pyrefly: ignore [missing-import]
import plotly.graph_objects as go

# ──────────────────────────────────────────────────────────────
# CẤU HÌNH TRANG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Job Trend IT – Nhóm 17",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<h1 style='text-align:center;color:#1E88E5;'>
📊 HỆ THỐNG PHÂN TÍCH VÀ DỰ ĐOÁN XU HƯỚNG VIỆC LÀM IT
</h1>
<p style='text-align:center;color:#555;'>
Đề tài 12 – Nhóm 17 | GVHD: ThS.NCS. Hà Thanh Dũng | Tuần 4
</p>
""", unsafe_allow_html=True)
st.markdown("---")

# ──────────────────────────────────────────────────────────────
# SỐ LIỆU CHÍNH THỨC – CHỐT TUẦN 4
# (Tất cả bảng/biểu đồ đều lấy từ đây – không được thay đổi)
# ──────────────────────────────────────────────────────────────
SO_LIEU_CHINH_THUC = {
    "Tổng mẫu": 3067,
    "Train": 2453,
    "Test": 614,
    "Seed": 42,
    "Metric chính": "Macro F1-score",
}

BANG_KETQUA = pd.DataFrame({
    "Mô hình": [
        "LR gốc (baseline T2)",
        "LR + class_weight balanced (T3)",
        "Linear SVM + balanced (T3)",
        "LR + title+description (T3 best)",
        "SVM + title+desc C=1.0 (T4 ★)",
    ],
    "Accuracy":    [0.5912, 0.6205, 0.7296, 0.7492, 0.8046],
    "Macro F1":    [0.3400, 0.6284, 0.7219, 0.7553, 0.7911],
    "Weighted F1": [0.5404, 0.6175, 0.7309, 0.7452, 0.8053],
    "∆ vs Baseline": ["—", "+0.2884", "+0.3819", "+0.4153", "+0.4511"],
    "Ghi chú": [
        "Không xử lý mất cân bằng",
        "Thêm class_weight=balanced",
        "SVM phù hợp TF-IDF thưa",
        "Title nhân đôi + description",
        "Kết hợp tốt nhất – Cải tiến T4",
    ],
})

# ──────────────────────────────────────────────────────────────
# LOAD DỮ LIỆU (thực hoặc demo)
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("../data/processed/job_cleaned.csv")
        # Kiểm tra cột cần thiết
        required = {"it_role_type", "city", "description"}
        if not required.issubset(df.columns):
            raise ValueError("Thiếu cột")
        df["location"] = df["city"].fillna("Không rõ")
        df["month"]    = "2025-01"   # Nếu không có cột ngày
        return df, False
    except Exception:
        # Demo mode – tạo dữ liệu giả lập
        np.random.seed(42)
        roles = [
            "Backend Developer", "Frontend Developer", "Full-stack Developer",
            "Software Engineer", "QA Engineer", "Mobile Developer",
            "AI/ML Engineer", "Java Developer", "Other IT Role",
        ]
        skills = ["Python", "Java", "React", "AWS", "SQL", "Docker", "NodeJS", "C#"]
        months = [f"2025-{m:02d}" for m in range(1, 13)]
        locs   = ["Hồ Chí Minh", "Hà Nội", "Đà Nẵng", "Cần Thơ", "Khác"]

        n = 3067
        data = {
            "it_role_type": np.random.choice(roles,  n, p=[.18,.12,.11,.10,.09,.08,.08,.06,.18]),
            "location":     np.random.choice(locs,   n, p=[.52,.30,.10,.05,.03]),
            "top_skill":    np.random.choice(skills, n),
            "salary_usd":   np.random.normal(1200, 350, n).clip(400, 3000).round(),
            "month":        np.random.choice(months, n),
        }
        return pd.DataFrame(data), True

df, is_demo = load_data()

if is_demo:
    st.warning(
        "⚠️ **Demo Mode:** Không tìm thấy `data/processed/job_cleaned.csv`. "
        "Dashboard đang dùng dữ liệu giả lập để minh họa. "
        "Đặt file dữ liệu đúng đường dẫn để xem kết quả thực.",
        icon="⚠️"
    )

# ──────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📈 Dashboard xu hướng việc làm",
    "📊 Đánh giá mô hình (Kết quả chính thức)",
    "🤖 Demo AI phân loại",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 – DASHBOARD XU HƯỚNG
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 🔍 Bộ lọc tương tác")
    cf1, cf2 = st.columns(2)
    with cf1:
        loc_opts = sorted(df["location"].unique())
        sel_loc  = st.multiselect("📍 Khu vực:", loc_opts, default=loc_opts)
    with cf2:
        role_opts = sorted(df["it_role_type"].unique())
        sel_role  = st.multiselect("💼 Vai trò:", role_opts, default=role_opts)

    fdf = df[df["location"].isin(sel_loc) & df["it_role_type"].isin(sel_role)]

    st.caption(f"Đang hiển thị **{len(fdf):,}** tin tuyển dụng"
               + (" (dữ liệu giả lập)" if is_demo else " (dữ liệu thực)"))
    st.markdown("---")

    # ── Hàng 1 ─────────────────────────────────────────────────
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown("**1. Top vai trò tuyển dụng nhiều nhất**")
        rc = fdf["it_role_type"].value_counts().reset_index()
        rc.columns = ["Vai trò", "Số lượng"]
        fig1 = px.bar(rc, x="Số lượng", y="Vai trò", orientation="h",
                      color="Số lượng", color_continuous_scale="Blues",
                      height=350)
        fig1.update_layout(yaxis={"categoryorder": "total ascending"},
                           margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig1, use_container_width=True)

    with r1c2:
        st.markdown("**2. Phân bố địa lý cơ hội việc làm**")
        lc = fdf["location"].value_counts().reset_index()
        lc.columns = ["Khu vực", "Số lượng"]
        fig2 = px.pie(lc, values="Số lượng", names="Khu vực",
                      hole=0.4, height=350,
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        fig2.update_layout(margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Hàng 2 ─────────────────────────────────────────────────
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown("**3. Xu hướng nhu cầu theo tháng**")
        if "month" in fdf.columns:
            trend = fdf.groupby(["month","it_role_type"]).size().reset_index(name="Số lượng")
            # Chỉ lấy top 5 vai trò cho dễ đọc
            top5  = fdf["it_role_type"].value_counts().head(5).index
            trend = trend[trend["it_role_type"].isin(top5)]
            fig3  = px.line(trend, x="month", y="Số lượng",
                            color="it_role_type", markers=True, height=350,
                            labels={"it_role_type":"Vai trò","month":"Tháng"})
            fig3.update_layout(margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig3, use_container_width=True)

    with r2c2:
        st.markdown("**4. Phân bố kỹ năng phổ biến**")
        if "top_skill" in fdf.columns:
            sc   = fdf["top_skill"].value_counts().reset_index()
            sc.columns = ["Kỹ năng", "Số lượng"]
            fig4 = px.bar(sc, x="Kỹ năng", y="Số lượng",
                          color="Số lượng", color_continuous_scale="Greens",
                          height=350)
            fig4.update_layout(margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig4, use_container_width=True)

    # ── Hàng 3: Lương (nếu có) ──────────────────────────────────
    if "salary_usd" in fdf.columns:
        st.markdown("**5. Tương quan kỹ năng và mức lương (USD)**")
        if "top_skill" in fdf.columns:
            fig5 = px.box(fdf, x="top_skill", y="salary_usd",
                          color="top_skill", height=380,
                          labels={"top_skill":"Kỹ năng","salary_usd":"Lương (USD)"},
                          color_discrete_sequence=px.colors.qualitative.Set2)
            fig5.update_layout(showlegend=False, margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig5, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 2 – KẾT QUẢ CHÍNH THỨC (SỐ LIỆU THỰC)
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📋 Kết quả thực nghiệm chính thức – Tuần 4")

    # Thông tin bộ số liệu
    mi1, mi2, mi3, mi4 = st.columns(4)
    mi1.metric("Tổng mẫu", f"{SO_LIEU_CHINH_THUC['Tổng mẫu']:,}")
    mi2.metric("Tập train", f"{SO_LIEU_CHINH_THUC['Train']:,}")
    mi3.metric("Tập test",  f"{SO_LIEU_CHINH_THUC['Test']:,}")
    mi4.metric("Random seed", str(SO_LIEU_CHINH_THUC['Seed']))

    st.info(
        f"**Metric chính: {SO_LIEU_CHINH_THUC['Metric chính']}** – "
        "Dữ liệu mất cân bằng (Other IT Role ~33%), dùng Accuracy sẽ thiên lệch về lớp đa số.",
        icon="ℹ️"
    )
    st.markdown("---")

    # Bảng kết quả – highlight mô hình tối ưu
    st.markdown("#### Bảng so sánh 5 mô hình (Tuần 2 → 3 → 4)")
    st.dataframe(
        BANG_KETQUA.style
            .highlight_max(subset=["Accuracy","Macro F1","Weighted F1"], color="#c8f7c5")
            .format({"Accuracy": "{:.4f}", "Macro F1": "{:.4f}", "Weighted F1": "{:.4f}"}),
        use_container_width=True,
        height=220,
    )
    st.success(
        "✅ **Mô hình tối ưu Tuần 4:** `SVM + title+description C=1.0`  \n"
        "Macro F1 tăng từ **0.3400** (baseline) → **0.7911** (+0.4511)"
    )
    st.markdown("---")

    # Biểu đồ so sánh Macro F1
    st.markdown("#### Biểu đồ so sánh Macro F1 – Tiến độ qua các tuần")
    fig_f1 = go.Figure()
    colors_bar = ["#ef9a9a","#ffcc80","#fff176","#a5d6a7","#42A5F5"]
    for i, row in BANG_KETQUA.iterrows():
        fig_f1.add_trace(go.Bar(
            x=[row["Mô hình"]], y=[row["Macro F1"]],
            marker_color=colors_bar[i],
            text=f"{row['Macro F1']:.4f}",
            textposition="outside",
            name=row["Mô hình"],
        ))
    fig_f1.update_layout(
        title="Macro F1-score – 5 mô hình (Metric chính, Test=614 mẫu, seed=42)",
        yaxis=dict(range=[0, 1.0], title="Macro F1"),
        xaxis_tickangle=-20,
        showlegend=False,
        height=420,
        margin=dict(l=0,r=0,t=50,b=100),
    )
    fig_f1.add_hline(y=0.75, line_dash="dash", line_color="gray",
                     annotation_text="Ngưỡng Khá (0.75)")
    fig_f1.add_hline(y=0.80, line_dash="dot",  line_color="navy",
                     annotation_text="Ngưỡng Tốt (0.80)")
    st.plotly_chart(fig_f1, use_container_width=True)

    # So sánh 3 metric
    st.markdown("#### So sánh Accuracy – Macro F1 – Weighted F1")
    fig_3m = go.Figure()
    x_vals = BANG_KETQUA["Mô hình"].tolist()
    fig_3m.add_trace(go.Bar(name="Accuracy",    x=x_vals, y=BANG_KETQUA["Accuracy"],    marker_color="#42A5F5"))
    fig_3m.add_trace(go.Bar(name="Macro F1 ★",  x=x_vals, y=BANG_KETQUA["Macro F1"],    marker_color="#EF5350"))
    fig_3m.add_trace(go.Bar(name="Weighted F1", x=x_vals, y=BANG_KETQUA["Weighted F1"], marker_color="#66BB6A"))
    fig_3m.update_layout(
        barmode="group", height=400,
        yaxis=dict(range=[0,1.0]),
        xaxis_tickangle=-15,
        margin=dict(l=0,r=0,t=20,b=120),
    )
    st.plotly_chart(fig_3m, use_container_width=True)

    # Cải tiến tuần 4
    st.markdown("#### Tóm tắt cải tiến Tuần 4")
    col_ci1, col_ci2 = st.columns(2)
    with col_ci1:
        st.markdown("""
**Đã thực hiện:**
- ✅ Kết hợp LinearSVC + title+description + C=1.0
- ✅ GridSearch C ∈ {0.1, 0.3, 0.5, 1.0, 2.0}
- ✅ Per-class F1 đầy đủ (yêu cầu thầy mục 10)
- ✅ Phân tích nhóm Other IT Role
- ✅ Ablation study (title vs desc vs kết hợp)
- ✅ Cross-validation 5-fold
- ✅ Đo inference time
        """)
    with col_ci2:
        st.markdown("""
**Kế hoạch Tuần 5:**
- → Thêm skill keywords (lang_list + tech_list)
- → Thử char n-gram cho C++, C#, .NET
- → Xem xét tách lớp khỏi Other IT Role
- → Hoàn thiện dashboard xu hướng kỹ năng theo thành phố
        """)

# ══════════════════════════════════════════════════════════════
# TAB 3 – DEMO AI PHÂN LOẠI (Yêu cầu thầy mục 9)
# "Nhập câu → dự đoán nhãn"
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🤖 Demo AI – Nhập mô tả → Dự đoán vai trò IT")
    st.caption("Mô hình: SVM + title+description C=1.0 | Macro F1=0.7911 | 11 lớp")
    st.markdown("---")

    # ── Load mô hình ──────────────────────────────────────────
    @st.cache_resource
    def load_model():
        """Thử load model thực (.pkl). Nếu không có → train nhanh từ CSV demo."""
        try:
            import joblib
            model   = joblib.load("models/svm_model.pkl")
            tfidf   = joblib.load("models/tfidf_vectorizer.pkl")
            classes = joblib.load("models/classes.pkl")
            return model, tfidf, classes, "real"
        except Exception:
            pass

        # Fallback: train trực tiếp từ file CSV nếu có
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.svm import LinearSVC
            from sklearn.model_selection import train_test_split

            df_t = pd.read_csv("../data/processed/job_cleaned.csv")
            df_t["it_role_type"] = df_t["it_role_type"].replace(
                {"Fullstack Developer": "Full-stack Developer"})
            dem = df_t["it_role_type"].value_counts()
            vt  = [r for r in dem.index if dem[r] >= 35 and r != "Other IT Role"]
            df_t["nhan"] = df_t["it_role_type"].apply(
                lambda x: x if x in vt else "Other IT Role")
            clean = lambda t: re.sub(r'\s+',' ',
                re.sub(r'[^\w\s.+#/]',' ', str(t).lower())).strip()
            df_t["tc"] = df_t["title"].apply(clean)
            df_t["dc"] = df_t["description"].apply(clean)
            df_t["X"]  = df_t["tc"] + " " + df_t["tc"] + " " + df_t["dc"]
            X_tr, _, y_tr, _ = train_test_split(
                df_t["X"], df_t["nhan"], test_size=0.2,
                random_state=42, stratify=df_t["nhan"])
            tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), sublinear_tf=True)
            clf   = LinearSVC(C=1.0, max_iter=3000, random_state=42, class_weight="balanced")
            clf.fit(tfidf.fit_transform(X_tr), y_tr)
            return clf, tfidf, list(clf.classes_), "trained"
        except Exception:
            return None, None, None, "demo"

    model, tfidf, classes, mode = load_model()

    if mode == "real":
        st.success("✅ Đã nạp mô hình thực từ file .pkl")
    elif mode == "trained":
        st.info("ℹ️ Mô hình được train trực tiếp từ dữ liệu khi khởi động app.")
    else:
        st.warning("⚠️ Chạy Demo Mode (không có dữ liệu/model) – kết quả là ví dụ minh họa.")

    # ── Giao diện nhập liệu ──────────────────────────────────
    st.markdown("#### Nhập thông tin tin tuyển dụng")
    col_inp, col_out = st.columns([3, 2])

    with col_inp:
        job_title = st.text_input(
            "📌 Tiêu đề công việc (Job Title):",
            placeholder="VD: Senior Backend Developer – Java Spring Boot",
        )
        job_desc = st.text_area(
            "📝 Mô tả công việc (Job Description):",
            height=200,
            placeholder=(
                "Dán nội dung mô tả vào đây...\n\n"
                "VD: We are looking for an experienced backend developer "
                "with strong Java skills. Build RESTful APIs using Spring Boot, "
                "design MySQL schemas, integrate Redis caching..."
            ),
        )

        # Nút mẫu nhanh
        st.markdown("**Hoặc chọn mẫu sẵn:**")
        c_s1, c_s2, c_s3, c_s4 = st.columns(4)
        if c_s1.button("Backend"):
            job_title = "Backend Developer Java"
            job_desc  = "Build RESTful APIs using Spring Boot MySQL Redis Docker CI/CD"
        if c_s2.button("Frontend"):
            job_title = "Frontend Developer ReactJS"
            job_desc  = "Build UI with ReactJS TypeScript Redux responsive design CSS"
        if c_s3.button("DevOps"):
            job_title = "DevOps Engineer"
            job_desc  = "CI/CD GitLab AWS Terraform Docker Kubernetes Prometheus monitoring"
        if c_s4.button("QA"):
            job_title = "QA Automation Engineer"
            job_desc  = "Selenium TestNG regression testing Jenkins JIRA Agile automation"

        predict_btn = st.button("🔍 Phân tích bằng AI", type="primary", use_container_width=True)

    with col_out:
        st.markdown("#### Kết quả dự đoán")

        if predict_btn:
            if not job_title.strip() and not job_desc.strip():
                st.warning("Vui lòng nhập tiêu đề hoặc mô tả công việc!")
            else:
                with st.spinner("Đang phân tích..."):
                    if model is not None and tfidf is not None:
                        # Dự đoán thực
                        clean = lambda t: re.sub(r'\s+',' ',
                            re.sub(r'[^\w\s.+#/]',' ', str(t).lower())).strip()
                        tc   = clean(job_title)
                        dc   = clean(job_desc)
                        x    = tc + " " + tc + " " + dc
                        vec  = tfidf.transform([x])
                        pred = model.predict(vec)[0]

                        # Top 3 (decision function)
                        scores  = model.decision_function(vec)[0]
                        top_idx = scores.argsort()[::-1][:3]
                        top3    = [(model.classes_[i], round(float(scores[i]),3))
                                   for i in top_idx]

                        st.success(f"**Vai trò dự đoán:**  \n# {pred}")
                        st.markdown("---")
                        st.markdown("**Top 3 ứng viên:**")
                        for rank, (role, score) in enumerate(top3, 1):
                            bar_w = max(0, min(100, int((score + 3) / 6 * 100)))
                            st.markdown(
                                f"`{rank}.` **{role}**  \n"
                                f"Score: `{score:+.3f}`"
                            )
                            st.progress(bar_w)

                    else:
                        # Demo mode – ví dụ minh họa
                        demo_pred = "Backend Developer"
                        demo_top3 = [
                            ("Backend Developer", 1.241),
                            ("Software Engineer", 0.823),
                            ("Java Developer",    0.512),
                        ]
                        st.success(f"**Vai trò dự đoán (Demo):**  \n# {demo_pred}")
                        st.caption("*Demo Mode – kết quả mẫu, không phản ánh dữ liệu thực*")
                        st.markdown("**Top 3 ứng viên:**")
                        for rank, (role, score) in enumerate(demo_top3, 1):
                            st.markdown(f"`{rank}.` **{role}** – score: `{score:+.3f}`")

        else:
            st.info("👆 Nhập thông tin bên trái rồi bấm **Phân tích bằng AI**")
            st.markdown("""
**Mô hình đang dùng:**
- Thuật toán: LinearSVC + class_weight balanced
- Đặc trưng: TF-IDF (title×2 + description)  
- Tham số: C=1.0, max_features=5000, ngram=(1,2)
- Macro F1: **0.7911** | Accuracy: **0.8046**
- Tập test: 614 mẫu | Seed=42
            """)

    # ── Hướng dẫn lưu model .pkl ──────────────────────────────
    with st.expander("📦 Hướng dẫn lưu model .pkl để app dùng model thực"):
        st.code("""
# Chạy trong notebook sau khi train xong:
import joblib, os
os.makedirs('models', exist_ok=True)
joblib.dump(m5,      'models/svm_model.pkl')
joblib.dump(tfidf_td,'models/tfidf_vectorizer.pkl')
joblib.dump(list(m5.classes_), 'models/classes.pkl')
print('✓ Đã lưu model')
        """, language="python")

# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "📌 Báo cáo Tuần 4 – Đồ án Ngôn ngữ lập trình Python | Nhóm 17 | "
    "Metric chính: Macro F1 | Số liệu: 3.067 mẫu, test=614, seed=42"
)
