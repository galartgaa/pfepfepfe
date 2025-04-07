import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Настройки отображения
st.set_page_config(layout="wide")
st.title("PFE для форвардного контракта")

# Загрузка данных
df = pd.read_csv('currencies_indicative_data.csv')

# Выбор инструмента
secid_list = df['secid'].unique()

selected_secid = st.selectbox("Выберите валютную пару (secid):", secid_list)

df_sec = df[df['secid'] == selected_secid].copy()
# Параметры симуляции
n_simulations = st.sidebar.slider("Количество итераций", 100, 10000, 1000, step=100)
n_days = st.sidebar.slider("Горизонт моделирования (дней)", 30, 952, 252, step=21)
confidence_level = st.sidebar.slider("Уровень доверия для PFE", 0.90, 0.99, 0.95)
# Расчет логарифмических доходностей
df_sec['log_return'] = np.log(df_sec['rate'] / df_sec['rate'].shift(1))
df_sec.dropna(inplace=True)

mu_daily = df_sec['log_return'].mean()
sigma_daily = df_sec['log_return'].std()
current_rate = df_sec['rate'].iloc[-1]

# Ввод форвардной цены
forward_price = st.number_input(
    "Введите форвардную цену",
    min_value=0.0,
    value=float(current_rate),
    step=0.01,
    format="%.4f"
)

# Симуляция одного пути
def simulate_path(current_price, mu, sigma, days):
    prices = np.zeros(days + 1)
    prices[0] = current_price
    random_returns = np.random.normal(mu, sigma, days)
    for t in range(1, days + 1):
        prices[t] = prices[t-1] * np.exp(random_returns[t-1])
    return prices

# Симуляция всех путей
np.random.seed(42)
all_paths = np.zeros((n_simulations, n_days + 1))
for i in range(n_simulations):
    all_paths[i, :] = simulate_path(current_rate, mu_daily, sigma_daily, n_days)

# MtM и PFE
mtm_values = all_paths - forward_price
pfe = np.zeros(n_days + 1)
epfe = np.zeros(n_days + 1)

for t in range(n_days + 1):
    mtm_t = mtm_values[:, t]
    positive_mtm = mtm_t[mtm_t > 0]
    if len(positive_mtm) > 0:
        pfe[t] = np.percentile(positive_mtm, confidence_level * 100)
        epfe[t] = np.mean(positive_mtm)

# EEPE
eepe = np.mean(epfe[1:])

# График путей
st.subheader("График путей")
fig1, ax1 = plt.subplots(figsize=(12, 5))
for i in range(min(100, n_simulations)):
    ax1.plot(all_paths[i, :], color='steelblue', alpha=0.1)

ax1.axhline(y=forward_price, color='red', linestyle='--', label=f'Форвард = {forward_price:.4f}')
ax1.plot(np.mean(all_paths, axis=0), color='black', linewidth=2, label='Средний путь')
ax1.set_title(f'Пути цен {selected_secid}')
ax1.set_xlabel("Дни")
ax1.set_ylabel("Курс")
ax1.legend()
ax1.grid(True, alpha=0.3)
st.pyplot(fig1)

# График PFE/EPFE
st.subheader("PFE и EPFE")
fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(pfe, label=f"PFE ({int(confidence_level*100)}%)", color='red')
ax2.plot(epfe, label="EPFE", color='blue')
ax2.fill_between(range(len(pfe)), pfe, alpha=0.2, color='red')
ax2.fill_between(range(len(epfe)), epfe, alpha=0.2, color='blue')
ax2.set_xlabel("Дни")
ax2.set_ylabel("Значение")
ax2.legend()
ax2.grid(True, alpha=0.3)
st.pyplot(fig2)

# Итоги
st.subheader("Результаты")
max_pfe = np.max(pfe)
max_pfe_day = np.argmax(pfe)
pfe_horizons = {
    '1 месяц (~21 день)': pfe[min(21, n_days)],
    '3 месяца (~63 дня)': pfe[min(63, n_days)],
    '6 месяцев (~126 дней)': pfe[min(126, n_days)],
    '9 месяцев (~189 дней)': pfe[min(189, n_days)],
    '1 год (~252 дня)': pfe[min(252, n_days)]
}

st.write(f"**Текущий спот-курс:** {current_rate:.4f}")
st.write(f"**Форвардная цена:** {forward_price:.4f}")
st.write(f"**Максимальное значение PFE:** {max_pfe:.4f} (на день {max_pfe_day})")
st.write(f"**Effective Expected Positive Exposure (EEPE):** {eepe:.4f}")

st.markdown("**PFE по горизонтам:**")
for k, v in pfe_horizons.items():
    st.write(f"- {k}: {v:.4f}")

