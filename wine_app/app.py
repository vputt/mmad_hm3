from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_curve


APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "model.joblib"
CONFIG_PATH = APP_DIR / "config.json"
TEST_PATH = APP_DIR / "test.csv"


@st.cache_resource  # загружаем модель один раз на весь сервер
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data  # читаем тестовую выборку один раз
def load_test_data():
    return pd.read_csv(TEST_PATH)


@st.cache_data  # вероятности не зависят от выбранного порога
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


# переносим значение ползунка в поле ввода
def update_number(feature):
    st.session_state[f"number_{feature}"] = st.session_state[f"slider_{feature}"]


# переносим введенное значение в ползунок
def update_slider(feature):
    st.session_state[f"slider_{feature}"] = st.session_state[f"number_{feature}"]


required_files = [MODEL_PATH, CONFIG_PATH, TEST_PATH]
missing_files = [path.name for path in required_files if not path.exists()]
if missing_files:
    st.error(f"Не найдены файлы: {', '.join(missing_files)}")
    st.stop()

model = load_model()
test = load_test_data()
features = list(model.feature_names_in_)
X_test = test[features]
y_test = test["target"]
probabilities = test_proba(model, X_test)

with CONFIG_PATH.open(encoding="utf-8") as file:
    config = json.load(file)

default_threshold = float(config["threshold"])

st.title("Сервис рекомендаций вин")
st.caption("Оценка качества вина с помощью логистической регрессии")

threshold = st.sidebar.slider("Порог классификации", min_value=0.0, max_value=1.0, value=default_threshold, step=0.01)
st.sidebar.write(f"Текущий порог: {threshold:.2f}")

single_tab, metrics_tab, batch_tab, coefficients_tab = st.tabs(["Одно вино", "Модельные метрики", "Прогноз по файлу", "Коэффициенты модели"])


with single_tab:
    st.subheader("Оценка одного вина")
    st.write("Выберите характеристики вина ползунком или введите точное значение")
    st.caption("Границы полей взяты из минимальных и максимальных значений test.csv")

    numeric_features = [feature for feature in features if feature != "is_red"]
    feature_steps = {
        "fixed_acidity": 0.1,
        "volatile_acidity": 0.001,
        "citric_acid": 0.01,
        "residual_sugar": 0.01,
        "chlorides": 0.001,
        "free_sulfur_dioxide": 0.5,
        "total_sulfur_dioxide": 0.5,
        "density": 0.00001,
        "pH": 0.01,
        "sulphates": 0.01,
        "alcohol": 0.1,
    }
    values = {}

    for feature in numeric_features:
        minimum = float(X_test[feature].min())
        maximum = float(X_test[feature].max())
        median = float(X_test[feature].median())
        step = feature_steps[feature]
        step_text = f"{step:.10f}".rstrip("0")
        decimal_places = len(step_text.split(".")[1]) if "." in step_text else 0
        value_format = f"%.{decimal_places}f"
        slider_key = f"slider_{feature}"
        number_key = f"number_{feature}"

        if slider_key not in st.session_state:
            st.session_state[slider_key] = median
            st.session_state[number_key] = median

        slider_column, number_column = st.columns([3, 1])
        with slider_column:
            st.slider(f"{feature}", min_value=minimum, max_value=maximum, step=step, format=value_format, key=slider_key, on_change=update_number, args=(feature,))
        with number_column:
            st.number_input("Точное значение", min_value=minimum, max_value=maximum, step=step, format=value_format, key=number_key, on_change=update_slider, args=(feature,))

        values[feature] = st.session_state[number_key]

    wine_color = st.radio("Цвет вина", ["Белое", "Красное"], horizontal=True)
    values["is_red"] = int(wine_color == "Красное")
    submitted = st.button("Оценить вино")

    if submitted:
        # собираем одну строку в порядке признаков модели
        wine = pd.DataFrame([[values[feature] for feature in features]], columns=features)
        probability = model.predict_proba(wine)[0, 1]
        prediction = probability >= threshold

        st.metric("Вероятность хорошего вина", f"{probability:.1%}")
        st.progress(float(probability))

        if prediction:
            st.success("Модель рекомендует это вино")
        else:
            st.warning("Модель не рекомендует это вино")


with metrics_tab:
    st.subheader("Метрики на тестовой выборке")

    # при смене порога меняется только итоговый класс
    predictions = probabilities >= threshold
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    first, second, third = st.columns(3)
    first.metric("Precision", f"{precision:.3f}")
    second.metric("Recall", f"{recall:.3f}")
    third.metric("F1", f"{f1:.3f}")

    st.write("Матрица ошибок")
    matrix_table = pd.DataFrame(matrix, index=["Фактически 0", "Фактически 1"], columns=["Предсказано 0", "Предсказано 1"])
    st.dataframe(matrix_table)

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    curve_precision, curve_recall, _ = precision_recall_curve(y_test, probabilities)

    tn, fp, fn, tp = matrix.ravel()
    current_fpr = fp / (fp + tn) if fp + tn else 0
    current_tpr = tp / (tp + fn) if tp + fn else 0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(fpr, tpr)
    axes[0].scatter(current_fpr, current_tpr, color="red")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set_title("ROC кривая")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].grid()

    axes[1].plot(curve_recall, curve_precision)
    axes[1].scatter(recall, precision, color="red")
    axes[1].set_title("PR кривая")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].grid()

    fig.tight_layout()
    st.pyplot(fig)

    st.info("Если увеличить порог, модель будет рекомендовать меньше вин, но рекомендации станут точнее\n\n" \
    "Если уменьшить порог, модель найдет больше хороших вин, но в подборку попадет больше посредственных")


with batch_tab:
    st.subheader("Пакетное предсказание")
    uploaded = st.file_uploader("CSV с характеристиками вин", type="csv")

    if uploaded is None:
        st.info("Загрузите CSV с теми же признаками, что использует модель")
    else:
        try:
            batch = pd.read_csv(uploaded)
            missing = [feature for feature in features if feature not in batch.columns]

            if missing:
                st.error(f"В файле нет колонок: {', '.join(missing)}")
            else:
                # берем признаки в том же порядке, что и при обучении
                batch_probabilities = model.predict_proba(batch[features])[:, 1]
                result = batch.copy()
                result["probability_good"] = batch_probabilities
                result["prediction"] = batch_probabilities >= threshold

                st.metric("Рекомендовано вин", int(result["prediction"].sum()))
                st.dataframe(result.sort_values("probability_good", ascending=False), hide_index=True)
                st.download_button("Скачать прогнозы", data=result.to_csv(index=False).encode("utf-8-sig"), file_name="wine_predictions.csv", mime="text/csv", on_click="ignore")
        except Exception as error:
            st.error(f"Не удалось обработать файл: {error}")


with coefficients_tab:
    st.subheader("Коэффициенты логистической регрессии")

    # коэффициенты относятся к признакам после масштабирования
    logistic_regression = model.named_steps["logisticregression"]
    coefficients = logistic_regression.coef_[0]
    coefficients_table = pd.DataFrame({"Признак": features, "Коэффициент": coefficients}).sort_values("Коэффициент")

    colors = np.where(coefficients_table["Коэффициент"] >= 0, "seagreen", "firebrick")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(coefficients_table["Признак"], coefficients_table["Коэффициент"], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Коэффициент")
    ax.set_title("Влияние признаков на прогноз")
    ax.grid(axis="x")
    fig.tight_layout()
    st.pyplot(fig)

    strongest_negative = coefficients_table.iloc[0]["Признак"]
    strongest_positive = coefficients_table.iloc[-1]["Признак"]

    st.write(f"Сильнее всего за хорошее вино: **{strongest_positive}**")
    st.write(f"Сильнее всего против хорошего вина: **{strongest_negative}**")
    st.caption("Положительный коэффициент повышает вероятность хорошего вина, отрицательный коэффициент понижает ее")
