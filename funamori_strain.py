import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Strain ε′₃₃ vs ψ — φ Sweep")

# --- Resolution control ---
st.subheader("Computation Settings")
col1, col2 = st.columns(2)
with col1:
    total_points = st.number_input("Total number of points (φ × ψ)", value=5000, min_value=10, step=10)
with col2:
    a_val = st.number_input("Lattice constant a (Å)", value=3.5)

# --- Inputs in compact layout ---
st.header("Input Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Miller Indices / Lattice")
    h = st.number_input("h", value=1.0)
    k = st.number_input("k", value=1.0)
    l = st.number_input("l", value=1.0)

with col2:
    st.subheader("Elastic Constants (GPa)")
    c11 = st.number_input("C11", value=450.0)
    c12 = st.number_input("C12", value=236.0)
    c44 = st.number_input("C44", value=42.5)

with col3:
    st.subheader("Stress Components (σᵢᵢ)")
    sigma_11 = st.number_input("σ₁₁", value=-0.5)
    sigma_22 = st.number_input("σ₂₂", value=-0.5)
    sigma_33 = st.number_input("σ₃₃", value=1)

# Compute square root to determine grid sizes
psi_steps = int(np.sqrt(total_points))
phi_steps = int(np.sqrt(total_points))

if st.button("Run Calculation"):
    # Elastic constants matrix
    elastic = np.array([
        [c11, c12, c12, 0, 0, 0],
        [c12, c11, c12, 0, 0, 0],
        [c12, c12, c11, 0, 0, 0],
        [0, 0, 0, c44, 0, 0],
        [0, 0, 0, 0, c44, 0],
        [0, 0, 0, 0, 0, c44]
    ])
    elastic_compliance = np.linalg.inv(elastic)

    sigma = np.array([
        [sigma_11, 0, 0],
        [0, sigma_22, 0],
        [0, 0, sigma_33]
    ])

    # Normalize direction
    H = h / a_val
    K = k / a_val
    L = l / a_val
    N = np.sqrt(K**2 + L**2)
    M = np.sqrt(H**2 + K**2 + L**2)

    phi_values = np.linspace(0, 2 * np.pi, phi_steps)
    psi_values = np.linspace(0, np.pi / 2, psi_steps)

    psi_list = []
    strain_list = []

    for phi in phi_values:
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        for psi in psi_values:
            cos_psi = np.cos(psi)
            sin_psi = np.sin(psi)

            # A (rotation wrt lab frame)
            A = np.array([
                [cos_phi * cos_psi, -sin_phi, cos_phi * sin_psi],
                [sin_phi * cos_psi, cos_phi, sin_phi * sin_psi],
                [-sin_psi, 0, cos_psi]
            ])

            # B (crystal orientation)
            B = np.array([
                [N/M, 0, H/M],
                [-H*K/(N*M), L/N, K/M],
                [-H*L/(N*M), -K/N, L/M]
            ])

            sigma_prime = A @ sigma @ A.T
            sigma_double_prime = B @ sigma_prime @ B.T

            # Strain tensor
            ε = np.zeros((3, 3))
            ε[0, 0] = elastic_compliance[0, 0] * sigma_double_prime[0, 0] + elastic_compliance[0, 1] * (sigma_double_prime[1, 1] + sigma_double_prime[2, 2])
            ε[1, 1] = elastic_compliance[1, 0] * sigma_double_prime[1, 1] + elastic_compliance[1, 1] * (sigma_double_prime[0, 0] + sigma_double_prime[2, 2])
            ε[2, 2] = elastic_compliance[2, 2] * sigma_double_prime[2, 2] + elastic_compliance[2, 0] * (sigma_double_prime[0, 0] + sigma_double_prime[1, 1])
            ε[0, 1] = ε[1, 0] = 0.5 * elastic_compliance[3, 3] * sigma_double_prime[0, 1]
            ε[0, 2] = ε[2, 0] = 0.5 * elastic_compliance[4, 4] * sigma_double_prime[0, 2]
            ε[1, 2] = ε[2, 1] = 0.5 * elastic_compliance[5, 5] * sigma_double_prime[1, 2]

            # ε'_33
            b13 = B[0, 2]
            b23 = B[1, 2]
            b33 = B[2, 2]

            strain_prime_33 = (
                b13**2 * ε[0, 0] +
                b23**2 * ε[1, 1] +
                b33**2 * ε[2, 2] +
                2 * b13 * b23 * ε[0, 1] +
                2 * b13 * b33 * ε[0, 2] +
                2 * b23 * b33 * ε[1, 2]
            )

            psi_list.append(np.degrees(psi))
            strain_list.append(strain_prime_33)

    st.success(f"Completed {phi_steps * psi_steps} evaluations.")

    # --- Plot result ---
    st.subheader("Scatter Plot: ε′₃₃ vs ψ")
    fig, ax = plt.subplots()
    scatter = ax.scatter(psi_list, strain_list, c=strain_list, cmap="viridis", s=1)
    ax.set_xlabel("ψ (degrees)")
    ax.set_ylabel("ε′₃₃")
    ax.set_xlim(0,90)
    ax.set_title("Strain ε′₃₃ vs ψ")
    ax.grid(True)
    cbar = fig.colorbar(scatter, ax=ax, label="ε′₃₃")
    st.pyplot(fig)