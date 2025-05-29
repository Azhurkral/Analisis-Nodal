import streamlit as st
import zipfile
from io import BytesIO
import pandas as pd
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import matplotlib.pyplot as plt
import math
import BeggsandBrill as BBe
from scipy.interpolate import interp1d

# === Funciones de conversión
stb_to_m3 = 0.1589873
m3_to_stb = 1 / stb_to_m3

def convert_units(oil_m3, water_m3, gor_m3m3, tht_C, twf_C, depth_m, Tsep_C):
    stb_per_m3 = 6.28981
    ft_per_m = 3.28084
    oil_stb = oil_m3 * stb_per_m3
    water_stb = water_m3 * stb_per_m3
    gor_cuft_bbl = gor_m3m3 * 5.61458
    tht_F = tht_C * 9 / 5 + 32
    twf_F = twf_C * 9 / 5 + 32
    depth_ft = depth_m * ft_per_m
    Tsep_F = Tsep_C * 9 / 5 + 32
    return oil_stb, water_stb, gor_cuft_bbl, tht_F, twf_F, depth_ft, Tsep_F

def temperature_gradient(T1, T2, Depth):
    return abs(T1 - T2) / Depth if Depth else 0

def pressure_traverse(oil_rate, water_rate, depths, temps, thp, GOR, gas_grav, oil_grav, wtr_grav,
                      diameter, angle):
    pressure_list = [thp]
    for i in range(1, len(depths)):
        dz = depths[i] - depths[i - 1]
        dpdz = BBe.Pgrad(pressure_list[i - 1], temps[i], oil_rate, water_rate, GOR, gas_grav,
                         oil_grav, wtr_grav, diameter, angle)
        pressure_list.append(round(pressure_list[i - 1] + dz * dpdz, 3))
    return pressure_list

def vlp_curve(rates_stb, depths, temps, thp, water_rate_stb, GOR, gas_grav, oil_grav, wtr_grav,
              diameter, angle):
    return [pressure_traverse(q, water_rate_stb, depths, temps, thp, GOR, gas_grav, oil_grav, wtr_grav,
                               diameter, angle)[-1] for q in rates_stb]

def ipr(pr, pb, ptest, qtest_m3d, wcut, re=1000, rw=0.25, h=100, ko=100, mu=100, Bo=1):
    qtest = qtest_m3d * m3_to_stb
    a = wcut * 0.8 + 0.2
    b = -0.8 * wcut + 0.8
    list_q = []
    list_pwf2 = []
    qob = 0

    if pb >= pr:
        qomax = qtest / (1 - a * ptest / pr - b * (ptest / pr) ** 2)
        for i in np.linspace(pr, 0, 50):
            qo = qomax * (1 - a * (i / pr) - b * (i / pr) ** 2)
            list_pwf2.append(i)
            list_q.append(round(qo * stb_to_m3, 2))
    else:
        for i in np.linspace(pr, 0, 50):
            list_pwf2.append(i)
            if ptest >= pb:
                j1 = qtest / (pr - ptest)
                qob = j1 * (pr - pb)
                qo = j1 * (pr - i) if i >= pb else qob + j1 * pb * (1 - a * i / pb - b * (i / pb) ** 2) / 1.8
            else:
                j1 = qtest / ((pr - pb) + pb / 1.8 * (1 - a * ptest / pr - b * (ptest / pr) ** 2))
                qob = j1 * (pr - pb)
                qo = j1 * (pr - i) if i >= pb else qob + j1 * pb / 1.8 * (1 - a * i / pb - b * (i / pb) ** 2)
            list_q.append(round(qo * stb_to_m3, 2))
        qomax = max(list_q)

    return list_q, list_pwf2, qob * stb_to_m3, qomax * stb_to_m3



def find_intersection(ipr_q, ipr_pwf, vlp_q, vlp_pwf):
    ipr_interp = interp1d(ipr_q, ipr_pwf, bounds_error=False, fill_value='extrapolate')
    vlp_interp = interp1d(vlp_q, vlp_pwf, bounds_error=False, fill_value='extrapolate')

    q_common = np.linspace(max(min(ipr_q[0], vlp_q[0]), 0), min(max(ipr_q[-1], vlp_q[-1]), 500), 1000)
    diff = ipr_interp(q_common) - vlp_interp(q_common)

    intersections = []

    for i in range(1, len(diff)):
        if diff[i-1] * diff[i] < 0:
            q1, q2 = q_common[i-1], q_common[i]
            p1_ipr, p2_ipr = ipr_interp(q1), ipr_interp(q2)
            p1_vlp, p2_vlp = vlp_interp(q1), vlp_interp(q2)

            m_ipr = (p2_ipr - p1_ipr) / (q2 - q1)
            m_vlp = (p2_vlp - p1_vlp) / (q2 - q1)
            m_diff = m_ipr - m_vlp

            if m_diff != 0:
                q_int = q1 + (p1_vlp - p1_ipr) / m_diff
                p_int = ipr_interp(q_int)
                intersections.append((float(q_int), float(p_int)))

    if intersections:
        # Selecciona el punto de cruce con mayor caudal
        q_final, p_final = max(intersections, key=lambda x: x[0])
        return round(q_final, 2), round(p_final, 2)
    return None, None

    ipr_interp = interp1d(ipr_q, ipr_pwf, bounds_error=False, fill_value='extrapolate')
    vlp_interp = interp1d(vlp_q, vlp_pwf, bounds_error=False, fill_value='extrapolate')

    # Dominio común de caudales
    q_common = np.linspace(max(min(ipr_q[0], vlp_q[0]), 0), min(max(ipr_q[-1], vlp_q[-1]), 500), 1000)

    # Diferencia entre curvas
    diff = ipr_interp(q_common) - vlp_interp(q_common)

    # Buscar cambio de signo
    for i in range(1, len(diff)):
        if diff[i-1] * diff[i] < 0:  # hubo cambio de signo => cruce
            # Interpolación lineal entre los dos puntos que rodean el cruce
            q1, q2 = q_common[i-1], q_common[i]
            p1_ipr, p2_ipr = ipr_interp(q1), ipr_interp(q2)
            p1_vlp, p2_vlp = vlp_interp(q1), vlp_interp(q2)

            # Interpolación lineal de punto de cruce
            m_ipr = (p2_ipr - p1_ipr) / (q2 - q1)
            m_vlp = (p2_vlp - p1_vlp) / (q2 - q1)
            m_diff = m_ipr - m_vlp

            if m_diff != 0:
                q_int = q1 + (p1_vlp - p1_ipr) / m_diff
                p_int = ipr_interp(q_int)
                return round(q_int, 2), round(p_int, 2)
    return None, None

    vlp_interp = interp1d(vlp_q, vlp_pwf, bounds_error=False, fill_value=np.nan)
    q_common = np.linspace(0, min(max(ipr_q), max(vlp_q)), 500)
    diff = np.abs(ipr_interp(q_common) - vlp_interp(q_common))
    idx = np.argmin(diff)
    if diff[idx] < 5:  # tolerancia
        return round(q_common[idx], 2), round(ipr_interp(q_common[idx]), 2)
    return None, None

# === Streamlit UI
st.title("Análisis Nodal - Curvas IPR y VLP")

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    st.subheader("Parámetros de Reservorio (IPR)")
    pr = st.number_input("Presión del reservorio (psi)", 500.0, 10000.0, 2000.0, 50.0)
    pb = st.number_input("Presión de burbuja (psi)", 500.0, pr, 1500.0, 50.0)
    ptest = st.number_input("Presión de prueba (psi)", 100.0, pr, 1500.0, 50.0)
    qtest = st.number_input("Caudal de prueba (m3/d)", 1.0, 2000.0, 100.0, 10.0)
    wcut = st.number_input("Corte de agua IPR (%)", 0.0, 100.0, 90.0, 1.0, key="wcut_ipr") / 100

with col2:
    st.subheader("Parámetros del Pozo (VLP)")
    q_total = st.number_input("Caudal de líquido bruto (m3/d)", 1.0, 500.0, 33.0)
    wcut_vlp = st.number_input("Corte de agua VLP (%)", 0.0, 100.0, 90.0, 1.0, key="wcut_vlp") / 100
    GOR = st.number_input("Razón gas/petróleo (m3/m3)", 0.0, 1000.0, 200.0)
    gas_grav = st.number_input("Gravedad específica del gas", 0.0, 2.0, 0.65)
    oil_grav = st.number_input("API del petróleo", 5.0, 50.0, 35.0)
    wtr_grav = st.number_input("Gravedad específica del agua", 0.0, 2.0, 1.07)
    diameter = st.number_input("Diámetro interno (pulgadas)", 0.5, 10.0, 2.441)
    angle = st.number_input("Ángulo del pozo (grados)", 0.0, 90.0, 90.0)
    thp = st.number_input("Presión en cabeza (psia)", 0.0, 5000.0, 150.0)
    tht = st.number_input("Temperatura cabeza (C)", 0.0, 200.0, 20.0)
    twf = st.number_input("Temperatura fondo (C)", 0.0, 300.0, 100.0)
    depth = st.number_input("Profundidad (m)", 100.0, 5000.0, 2000.0)
    roughness = st.number_input("Rugosidad (pulgadas)", 0.00001, 0.01, 0.000059, format="%.10f")
    Psep = st.number_input("Presión del separador (psia)", 0.0, 1000.0, 114.7)
    Tsep = st.number_input("Temperatura del separador (C)", 0.0, 150.0, 50.0)

# Calcular caudales para VLP a partir de bruta y wcut
water_rate = q_total * wcut_vlp
oil_rate = q_total * (1 - wcut_vlp)

# Conversión y cálculo VLP
oil_stb, water_stb, GOR_cuft, tht_F, twf_F, depth_ft, Tsep_F = convert_units(
    oil_rate, water_rate, GOR, tht, twf, depth, Tsep
)
depths = np.linspace(0, depth_ft, 50)
t_grad = temperature_gradient(tht_F, twf_F, depth_ft)
temps = tht_F + t_grad * depths

rate_array_m3 = np.arange(2.5, 200.01, 2.5)
rate_array_stb = rate_array_m3 * 6.28981
vlp_bhp = vlp_curve(rate_array_stb, depths, temps, thp, water_stb, GOR_cuft, gas_grav, oil_grav,
                    wtr_grav, diameter, angle)

# Cálculo IPR
ipr_q, ipr_pwf, qob_m3, qomax_m3 = ipr(pr, pb, ptest, qtest, wcut)

# Encontrar intersección entre curvas
q_int, p_int = find_intersection(ipr_q, ipr_pwf, rate_array_m3, vlp_bhp)

# Gráfico y resultados
with col3:
    st.subheader("Curvas IPR y VLP")
    fig, ax = plt.subplots()
    ax.plot(ipr_q, ipr_pwf, label="IPR", marker="x")
    ax.plot(rate_array_m3, vlp_bhp, label="VLP", marker="o")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Caudal (m3/d)")
    ax.set_ylabel("Presión de fondo (psi)")
    ax.set_title("Curvas IPR y VLP")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()
    if q_int and p_int:
        ax.plot(q_int, p_int, marker='o', color='red')
        ax.annotate(f"Intersección\nQ={q_int} m3/d\nP={p_int} psi", (q_int, p_int), textcoords="offset points", xytext=(10,10), ha='left')
    st.pyplot(fig)

    st.markdown("### Resultados Calculados (IPR)")
    st.write(f"Caudal máximo estimado (Qo_max): **{round(qomax_m3, 2)} m3/d**")
    st.write(f"Caudal al punto de burbuja (Qo@Pb): **{round(qob_m3, 2)} m3/d**")
    if q_int and p_int:
        st.write(f"Punto de intersección entre IPR y VLP: **Q = {q_int} m3/d**, **Pwf = {p_int} psi**")

# === Mostrar tablas interactivas ===
st.write("\n\n")
st.subheader("Tablas de datos")

# Tabla IPR
if st.checkbox("Mostrar tabla IPR"):
    df_ipr = pd.DataFrame({
        "Caudal IPR (m3/día)": ipr_q,
        "Presión en fondo IPR (psi)": ipr_pwf
    })
    st.dataframe(df_ipr)

# Tabla VLP
if st.checkbox("Mostrar tabla VLP"):
    df_vlp = pd.DataFrame({
        "Caudal VLP (m3/día)": rate_array_m3,
        "Presión en fondo VLP (psi)": vlp_bhp
    })
    st.dataframe(df_vlp)

# Tabla de Inputs
if st.checkbox("Mostrar parámetros de entrada"):
    inputs_dict = {
        "Presión del reservorio (psi)": pr,
        "Presión de burbuja (psi)": pb,
        "Presión de prueba (psi)": ptest,
        "Caudal de prueba (m3/d)": qtest,
        "Corte de agua IPR (%)": wcut * 100,
        "Caudal total (m3/d)": q_total,
        "Corte de agua VLP (%)": wcut_vlp * 100,
        "GOR (m3/m3)": GOR,
        "Gravedad gas": gas_grav,
        "API petróleo": oil_grav,
        "Gravedad agua": wtr_grav,
        "Diámetro (pulgadas)": diameter,
        "Ángulo (grados)": angle,
        "THP (psia)": thp,
        "THT (°C)": tht,
        "TWF (°C)": twf,
        "Profundidad (m)": depth
    }
    st.dataframe(pd.DataFrame(list(inputs_dict.items()), columns=["Parámetro", "Valor"]))
