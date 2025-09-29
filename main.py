import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np

data = [
    ["Toyota", "Corolla", "Sedán", "Gasolina", 16, 23000, 470, "FWD", "Intermedia"],
    ["Ford", "Ranger", "Pickup", "Diesel", 10, 38000, 900, "4x4", "Intermedia"],
    ["Tesla", "Model 3", "Sedán", "Eléctrico", 18, 42000, 425, "RWD", "Alta"],
    ["Chevrolet", "Spark", "Hatchback", "Gasolina", 19, 13500, 170, "FWD", "Básica"],
    ["Hyundai", "Tucson", "SUV", "Gasolina", 13, 29000, 620, "AWD", "Intermedia"],
    ["Kia", "Rio", "Sedán", "Gasolina", 17, 16000, 390, "FWD", "Básica"],
    ["BMW", "X5", "SUV", "Diesel", 9, 65000, 650, "AWD", "Alta"],
    ["Audi", "A4", "Sedán", "Gasolina", 12, 46000, 480, "AWD", "Alta"],
    ["Nissan", "Leaf", "Hatchback", "Eléctrico", 16, 33000, 380, "FWD", "Intermedia"],
    ["Jeep", "Wrangler", "SUV", "Gasolina", 8, 44000, 900, "4x4", "Alta"],
    ["Honda", "Civic", "Sedán", "Gasolina", 15, 24000, 450, "FWD", "Intermedia"],
    ["Mazda", "CX-5", "SUV", "Gasolina", 12, 31000, 550, "AWD", "Intermedia"],
    ["Renault", "Duster", "SUV", "Diesel", 14, 22000, 475, "FWD", "Básica"],
    ["Mercedes-Benz", "GLC", "SUV", "Gasolina", 11, 57000, 600, "AWD", "Alta"],
    ["Volkswagen", "Golf", "Hatchback", "Híbrido", 18, 25000, 380, "FWD", "Intermedia"]
]

df = pd.DataFrame(data, columns=["Marca","Modelo","Tipo","Combustible","Consumo","Precio","Maletero","Tracción","Tecnología"])

st.title("🔍 Recomendador de Autos con KNN")

# ====== INPUTS ======
tipo = st.selectbox("Tipo de auto", df["Tipo"].unique())
combustible = st.selectbox("Combustible", df["Combustible"].unique())
consumo = st.slider("Consumo mínimo (km/L)", 0, 20, 14)
precio_min, precio_max = st.slider(
    "Rango de precio deseado ($)",
    min_value=10000,
    max_value=70000,
    value=(15000, 30000),
    step=1000
)
maletero = st.number_input("Capacidad mínima del maletero (L)", value=400)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("🚗 **Consumo (km/L)**")
    peso_consumo = st.slider(" ", 0.1, 5.0, 1.0, 0.1, key="consumo")
with col2:
    st.markdown("💰 **Precio**")
    peso_precio = st.slider(" ", 0.1, 5.0, 1.0, 0.1, key="precio")
with col3:
    st.markdown("🎒 **Maletero (L)**")
    peso_maletero = st.slider(" ", 0.1, 5.0, 1.0, 0.1, key="maletero")

st.markdown("### 📊 Tus prioridades")
st.info(f"""
    - 🚗 Consumo: **{peso_consumo}**
    - 💰 Precio: **{peso_precio}**
    - 🎒 Maletero: **{peso_maletero}**
    """)

pesos = np.array([peso_consumo, peso_precio, peso_maletero])
k = st.slider("Número de recomendaciones (k)", min_value=1, max_value=5, value=3)

# ====== FILTRO OBLIGATORIO ======
df_filtrado = df[
    (df["Tipo"] == tipo) &
    (df["Combustible"] == combustible) &
    (df["Precio"] >= precio_min) &
    (df["Precio"] <= precio_max)
]

if st.button("Recomendar"):
    if df_filtrado.empty:
        st.error("❌ No se encontraron autos que cumplan tus filtros obligatorios.")
    else:
        # ====== NORMALIZACIÓN ======
        scaler = MinMaxScaler()
        df_scaled = df_filtrado.copy()
        df_scaled[["Consumo","Precio","Maletero"]] = scaler.fit_transform(
            df_filtrado[["Consumo","Precio","Maletero"]]
        )

        usuario_vector = np.array([[consumo, (precio_min+precio_max)/2, maletero]])
        usuario_vector = scaler.transform(usuario_vector)

        # Aplicar pesos
        autos_vector = df_scaled[["Consumo","Precio","Maletero"]].values * pesos
        usuario_vector = usuario_vector * pesos

        # Ajuste de k
        k_ajustado = min(k, len(df_scaled))
        if k_ajustado < k:
            st.info(f"ℹ️ Solo hay {len(df_scaled)} autos disponibles, se ajustó k = {k_ajustado}")

        # ====== KNN ======
        knn = NearestNeighbors(n_neighbors=k_ajustado, metric="euclidean")
        knn.fit(autos_vector)
        distancias, indices = knn.kneighbors(usuario_vector)

        # ====== CALCULAR SIMILITUD CORRECTA ======
        similitudes = 1 / (1 + distancias[0])

        df_recomendados = df_filtrado.iloc[indices[0]].copy()
        df_recomendados["Similitud"] = similitudes

        st.success("✅ Aquí están tus recomendaciones:")
        for i, row in df_recomendados.sort_values(by="Similitud", ascending=False).iterrows():
            st.markdown(f"""
            <div style="background-color:#f9f9f9; padding:15px; border-radius:15px; margin-bottom:15px; 
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <h3 style="color:#2c3e50;">🚘 {row['Marca']} {row['Modelo']}</h3>
                <ul style="list-style:none; padding-left:0;">
                    <li><b>Tipo:</b> {row['Tipo']}</li>
                    <li><b>Combustible:</b> {row['Combustible']}</li>
                    <li><b>Consumo:</b> {row['Consumo']} km/L</li>
                    <li><b>Precio:</b> ${row['Precio']:,}</li>
                    <li><b>Maletero:</b> {row['Maletero']} L</li>
                    <li><b>Tracción:</b> {row['Tracción']}</li>
                    <li><b>Tecnología:</b> {row['Tecnología']}</li>
                </ul>
                <p><b>🔗 Similitud:</b> {row['Similitud']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

