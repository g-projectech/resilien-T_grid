import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import styles

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & REFINED THEME (DEEP SLATE)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Terna Grid Resilience AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich high-fidelity engineering styling
styles.apply_base_styles()

# ---------------------------------------------------------
# 2. GRID TOPOLOGY INITIALIZATION (G = V, E) (12 NODES)
# ---------------------------------------------------------
@st.cache_data
def generate_terna_grid():
    """
    Initializes a NetworkX Graph representing Terna's regional power grid topology.
    Includes 12 nodes with spatial positions and slope attributes.
    """
    G = nx.Graph()
    
    # 12 nodes representing generators (plants), substations, and critical transmission towers
    # Slope is terrain inclination in degrees (susceptibility factor)
    # pos is 2D coordinates for physical grid rendering
    nodes_data = {
        "Centrale Alpina (Gen-1)": {"pos": (1.0, 5.0), "type": "plant", "slope": 12.0},
        "Traliccio 101": {"pos": (2.2, 4.8), "type": "tower", "slope": 42.0},
        "Traliccio 102": {"pos": (3.5, 4.5), "type": "tower", "slope": 35.0},
        "Sottostazione Val Camonica (Sub-1)": {"pos": (5.0, 4.6), "type": "substation", "slope": 15.0},
        "Traliccio 103 (Hub)": {"pos": (3.8, 3.2), "type": "tower", "slope": 22.0},
        "Traliccio 104": {"pos": (2.0, 2.5), "type": "tower", "slope": 48.0},
        "Sottostazione Valtellina (Sub-2)": {"pos": (4.8, 2.0), "type": "substation", "slope": 14.0},
        "Traliccio 105": {"pos": (5.8, 3.3), "type": "tower", "slope": 38.0},
        "Sottostazione Garda (Sub-3)": {"pos": (6.5, 1.8), "type": "substation", "slope": 10.0},
        "Traliccio 106": {"pos": (5.5, 1.0), "type": "tower", "slope": 28.0},
        "Traliccio 107": {"pos": (2.5, 3.8), "type": "tower", "slope": 30.0},
        "Centrale Termoelettrica (Gen-2)": {"pos": (0.8, 2.8), "type": "plant", "slope": 11.0},
    }
    
    for name, attrs in nodes_data.items():
        G.add_node(name, **attrs)
        
    # Grid transmission lines (Edges)
    edges = [
        ("Centrale Alpina (Gen-1)", "Traliccio 101"),
        ("Traliccio 101", "Traliccio 102"),
        ("Traliccio 102", "Sottostazione Val Camonica (Sub-1)"),
        ("Sottostazione Val Camonica (Sub-1)", "Traliccio 105"),
        ("Traliccio 105", "Sottostazione Garda (Sub-3)"),
        ("Centrale Alpina (Gen-1)", "Traliccio 107"),
        ("Traliccio 107", "Traliccio 103 (Hub)"),
        ("Traliccio 103 (Hub)", "Sottostazione Val Camonica (Sub-1)"),
        ("Traliccio 103 (Hub)", "Sottostazione Valtellina (Sub-2)"),
        ("Traliccio 103 (Hub)", "Traliccio 104"),
        ("Traliccio 104", "Centrale Termoelettrica (Gen-2)"),
        ("Centrale Termoelettrica (Gen-2)", "Traliccio 106"),
        ("Traliccio 106", "Sottostazione Valtellina (Sub-2)"),
        ("Sottostazione Valtellina (Sub-2)", "Sottostazione Garda (Sub-3)"),
    ]
    G.add_edges_from(edges)
    return G

G = generate_terna_grid()
node_list = list(G.nodes())
N = len(node_list)
node_to_idx = {node: i for i, node in enumerate(node_list)}


# ---------------------------------------------------------
# 3. SIDEBAR: PREDICTIVE FORCING CONTROLS
# ---------------------------------------------------------

# Translation mapping for scenarios and UI text
from translations import time_options, node_display_names, texts

# Configurator elements placeholders
config_title_placeholder = st.sidebar.empty()
config_desc_placeholder = st.sidebar.empty()

# Selettore di Lingua / Language Switcher
st.sidebar.markdown("---")
st.sidebar.markdown("🌐 **LINGUA / LANGUAGE**")
lang = st.sidebar.selectbox("Seleziona la lingua / Select language", ["🇮🇹 Italiano", "🇬🇧 English"])
st.sidebar.markdown("---")

lang_code = "it" if "Italiano" in lang else "en"

# Populate configurator text based on selected language
config_title_placeholder.markdown(texts[lang_code]["configurator_title"], unsafe_allow_html=True)
config_desc_placeholder.markdown(texts[lang_code]["configurator_desc"])

# Climatic Simulation Slider
st.sidebar.markdown(texts[lang_code]["weather_evolution"])
time_key = st.sidebar.select_slider(
    texts[lang_code]["weather_slider"],
    options=["t_0", "t_12", "t_24", "t_48"],
    format_func=lambda x: time_options[lang_code][x]
)

# InSAR Interferometric Anomaly Injector
st.sidebar.markdown("---")
st.sidebar.markdown(texts[lang_code]["insar_injector"])
injected_node = st.sidebar.selectbox(
    texts[lang_code]["select_component"],
    node_list,
    index=5,
    format_func=lambda x: node_display_names[lang_code][x]
) # Default to Traliccio 104
injected_val = st.sidebar.slider(
    texts[lang_code]["ground_displacement"],
    0.0,
    25.0,
    18.5,
    help=texts[lang_code]["ground_displacement_help"]
)


# ---------------------------------------------------------
# 4. CORE MATHEMATICAL ENGINE (GCN & TOPOLOGY COUPLING)
# ---------------------------------------------------------

# -- Mathematical Step 1: Self-connected Adjacency --
# Equation: A_tilde = A + I
A = nx.to_numpy_array(G, nodelist=node_list) # Adjacency Matrix A (N x N)
I = np.eye(N)                                 # Identity Matrix I (N x N)
A_tilde = A + I                              # Self-connected Adjacency A_tilde (N x N)

# -- Mathematical Step 2: Symmetric Laplacian Normalization --
# Equation: A_hat = D_tilde^{-1/2} A_tilde D_tilde^{-1/2}
d_tilde = np.sum(A_tilde, axis=1)                          # Degree vector for each node (N,)
D_tilde_inv_sqrt = np.diag(1.0 / np.sqrt(d_tilde))          # D_tilde^{-1/2} Diagonal Matrix (N x N)
A_hat = D_tilde_inv_sqrt @ A_tilde @ D_tilde_inv_sqrt       # Symmetric normalized adjacency matrix (N x N)

# -- Feature Matrix X Generation --
# Gaussian spatial distribution of rainfall representing a moving storm front
if time_key == "t_0":
    x_c, y_c = 0.5, 4.0
    R_max = 15.0
    sigma_x, sigma_y = 2.0, 2.0
elif time_key == "t_12":
    x_c, y_c = 2.2, 3.8
    R_max = 65.0
    sigma_x, sigma_y = 2.5, 2.5
elif time_key == "t_24":
    x_c, y_c = 3.8, 3.2
    R_max = 140.0
    sigma_x, sigma_y = 3.0, 3.0 # Raggio d'azione notevolmente allargato
else: # t_48
    x_c, y_c = 4.5, 3.0
    R_max = 220.0
    sigma_x, sigma_y = 3.8, 3.8 # Copertura quasi totale della griglia topologica

scenario_desc = texts[lang_code]["scenarios"][time_key]

# Compute Rainfall feature per node using primary AND secondary storm cells
rainfalls = {}
for node in G.nodes():
    x, y = G.nodes[node]["pos"]
    
    # Calcolo intensità cella temporalesca principale
    dist_sq_1 = ((x - x_c) ** 2) / (2 * sigma_x ** 2) + ((y - y_c) ** 2) / (2 * sigma_y ** 2)
    rain = R_max * np.exp(-dist_sq_1)
    
    # Iniezione Cella Temporalesca Secondaria (Caos Aggiuntivo) per scenari critici
    if time_key in ["t_24", "t_48"]:
        x_c2, y_c2 = 2.0, 4.5 # Colpisce i cluster di tralicci a Nord-Ovest
        dist_sq_2 = ((x - x_c2) ** 2) / (2 * 2.2 ** 2) + ((y - y_c2) ** 2) / (2 * 2.2 ** 2)
        rain += (R_max * 0.75) * np.exp(-dist_sq_2) # Aggiunge il 75% di pioggia dalla seconda cella
        
    rainfalls[node] = float(np.round(max(rain, 3.0), 2)) # Assure background noise

# Generate InSAR displacements (Baseline random creep + manual anomaly injection)
np.random.seed(42)
insar_displacements = {node: float(np.round(np.random.uniform(0.5, 2.5), 2)) for node in G.nodes()}
insar_displacements[injected_node] = float(np.round(injected_val, 2))

# Assemble Raw Feature Matrix X_raw (N x 3)
# Columns: [Rainfall (mm), InSAR (mm/month), Slope (degrees)]
X_raw = np.zeros((N, 3))
for i, node in enumerate(node_list):
    X_raw[i, 0] = rainfalls[node]
    X_raw[i, 1] = insar_displacements[node]
    X_raw[i, 2] = G.nodes[node]["slope"]

# Normalize features to [0, 1] for stable GCN computation (Rainfall max=200, InSAR max=25, Slope max=50)
X = np.zeros((N, 3))
X[:, 0] = X_raw[:, 0] / 200.0  # Feature 1: Rainfall
X[:, 1] = X_raw[:, 1] / 25.0   # Feature 2: InSAR
X[:, 2] = X_raw[:, 2] / 50.0   # Feature 3: Slope

# -- Mathematical Step 3: GCN Layer Forward Pass --
# Equation: Z = sigmoid( A_hat X W + b )
# Weight vector W (3 x 1) optimization parameter prioritizing: InSAR (2.5) > Slope (1.8) > Rainfall (1.5)
W = np.array([[1.5], 
              [2.5], 
              [1.8]]) 
bias = -2.2 # Bias value to center sigmoid trigger threshold

H_local = X @ W                     # Linear feature projection (N x 1)
H_conv = A_hat @ H_local            # Graph spectral message passing/convolution (N x 1)
Z_gcn = 1.0 / (1.0 + np.exp(-(H_conv + bias))) # Sigmoid activation (N x 1)

# -- Mathematical Step 4: Physics-Informed Centrality Coupling --
# Equation: SRI_i = (w1*Z_i + w2*FS_vuln_i) * C_{topo, i}
betweenness = nx.betweenness_centrality(G) # NetworkX Exact Betweenness Centrality
max_bc = max(betweenness.values()) if max(betweenness.values()) > 0 else 1.0

# Scale Betweenness Centrality to [0.3, 1.0] representing topological threat multiplier
centrality_factors = {node: 0.3 + 0.7 * (betweenness[node] / max_bc) for node in G.nodes()}

sri = {}
for i, node in enumerate(node_list):
    # Recupera i dati fisici reali per il nodo
    rain = X_raw[i, 0]
    insar = X_raw[i, 1]
    slope = X_raw[i, 2]
    
    # Modello Geotecnico (Infinite Slope Stability)
    c_prime = 15.0 * np.exp(-0.08 * insar)
    u_pore = 0.15 * rain
    sigma_normal = 50.0
    phi_rad = np.radians(32.0)
    slope_rad = np.radians(slope)
    
    numerator = c_prime + (sigma_normal - u_pore) * np.tan(phi_rad)
    denominator = sigma_normal * np.tan(slope_rad)
    fs = numerator / denominator if denominator > 0 else 5.0
    
    # La vulnerabilità fisica aumenta drasticamente se il Fattore di Sicurezza (FS) scende
    geotech_vulnerability = 1.0 if fs < 1.0 else (max(0.0, 1.3 - fs) / 0.3 if fs < 1.3 else 0.0)
    
    # SRI finale ponderato: AI (GCN) 60% + Fisica Geotecnica 40%, moltiplicato per la Centralità
    combined_risk = (Z_gcn[i, 0] * 0.6) + (geotech_vulnerability * 0.4)
    sri[node] = float(np.round(combined_risk * centrality_factors[node], 4))


# ---------------------------------------------------------
# 5. HEADER BLOCK & SYSTEM MONITORING CARD
# ---------------------------------------------------------
styles.render_header(lang=lang_code)

# Grid Resilience State calculations
threshold = 0.75
critical_nodes = [node for node in node_list if sri[node] > threshold]
num_critical = len(critical_nodes)
avg_sri = float(np.mean(list(sri.values())))

# Display KPIs
kpi_class = "kpi-val-green" if num_critical == 0 else ("kpi-val-amber" if num_critical < 3 else "kpi-val-red")
kpi_status = (
    texts[lang_code]["kpi_stable"]
    if num_critical == 0
    else (texts[lang_code]["kpi_warning"] if num_critical < 3 else texts[lang_code]["kpi_critical"])
)

styles.render_kpis(kpi_class, kpi_status, num_critical, N, threshold, avg_sri, lang=lang_code)

time_horizon_label = time_options[lang_code][time_key]
st.info(f"📋 **{texts[lang_code]['active_scenario']} ({time_horizon_label}):** {scenario_desc}")


# ---------------------------------------------------------
# 6. MAIN WORKSPACE Scientific Tabs
# ---------------------------------------------------------
tab1, tab2 = st.tabs([texts[lang_code]["tab_spectral"], texts[lang_code]["tab_mitigation"]])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(texts[lang_code]["map_title"])
        st.markdown(texts[lang_code]["map_desc"], unsafe_allow_html=True)
        
        # Build coordinates and edge lists
        edge_normal_x = []
        edge_normal_y = []
        edge_compromised_x = []
        edge_compromised_y = []
        
        for u, v in G.edges():
            x0, y0 = G.nodes[u]["pos"]
            x1, y1 = G.nodes[v]["pos"]
            
            # Cascading Edge State: if either node exceeds SRI threshold, the edge is compromised/overloaded
            if sri[u] > threshold or sri[v] > threshold:
                edge_compromised_x.extend([x0, x1, None])
                edge_compromised_y.extend([y0, y1, None])
            else:
                edge_normal_x.extend([x0, x1, None])
                edge_normal_y.extend([y0, y1, None])
                
        # Create normal edge trace (Slate gray)
        trace_edges_normal = go.Scatter(
            x=edge_normal_x, y=edge_normal_y,
            line=dict(width=1.5, color='#455A64'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create compromised/downstream edge trace (Dashed Fracture Red)
        trace_edges_compromised = go.Scatter(
            x=edge_compromised_x, y=edge_compromised_y,
            line=dict(width=3, color='#FF1744', dash='dash'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Build node trace details
        node_x = [G.nodes[n]["pos"][0] for n in node_list]
        node_y = [G.nodes[n]["pos"][1] for n in node_list]
        node_colors = [sri[n] for n in node_list]
        node_sizes = [20 + 35 * centrality_factors[n] for n in node_list]
        
        node_type_mapping = {
            "plant": texts[lang_code]["node_type_plant"],
            "tower": texts[lang_code]["node_type_tower"],
            "substation": texts[lang_code]["node_type_substation"]
        }
        hover_texts = []
        for n in node_list:
            idx = node_to_idx[n]
            hover_texts.append(
                f"<b>{node_display_names[lang_code][n]}</b><br>"
                f"{texts[lang_code]['hover_type']}: {node_type_mapping[G.nodes[n]['type']]}<br>"
                f"{texts[lang_code]['hover_sri']}: {sri[n]:.3f}<br>"
                f"  ├─ {texts[lang_code]['hover_gcn']}: {Z_gcn[idx, 0]:.3f}<br>"
                f"  └─ {texts[lang_code]['hover_centrality']}: {betweenness[n]:.3f}<br>"
                f"{texts[lang_code]['hover_slope']}: {G.nodes[n]['slope']}°<br>"
                f"{texts[lang_code]['hover_rainfall']}: {rainfalls[n]:.1f} mm<br>"
                f"{texts[lang_code]['hover_insar']}: {insar_displacements[n]:.2f} {texts[lang_code]['month_unit']}"
            )
            
        trace_nodes = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=[node_display_names[lang_code][n].split(" (")[0] for n in node_list],
            textposition="top center",
            hoverinfo='text',
            hovertext=hover_texts,
            textfont=dict(color='#ECEFF1', size=11, family='Outfit, sans-serif'),
            marker=dict(
                showscale=True,
                colorscale=[[0.0, '#00E676'], [0.5, '#FFB300'], [1.0, '#FF1744']],
                cmin=0.0,
                cmax=1.0,
                color=node_colors,
                size=node_sizes,
                line=dict(width=1.5, color='#0E1117'),
                colorbar=dict(
                    title=dict(text=texts[lang_code]["colorbar_title"], side="right"),
                    thickness=15,
                    outlinecolor="#2D3748",
                    tickfont=dict(color="#ECEFF1")
                )
            )
        )
        
        # Assemble Plotly Figure
        fig = go.Figure(
            data=[trace_edges_normal, trace_edges_compromised, trace_nodes],
            layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=10, l=10, r=10, t=10),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0.3, 7.2]),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0.4, 5.6]),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=550
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.markdown(texts[lang_code]["risk_ranking_title"])
        
        col_comp = texts[lang_code]["df_col_component"]
        col_sri = texts[lang_code]["df_col_sri"]
        col_gcn = texts[lang_code]["df_col_gcn"]
        col_bc = texts[lang_code]["df_col_centrality"]

        # Build Pandas DataFrame for visual analytics
        df_risk = pd.DataFrame({
            col_comp: [node_display_names[lang_code][n].split(" (")[0] for n in node_list],
            col_sri: [sri[n] for n in node_list],
            col_gcn: [float(Z_gcn[node_to_idx[n], 0]) for n in node_list],
            col_bc: [betweenness[n] for n in node_list]
        }).sort_values(by=col_sri, ascending=False)
        
        st.dataframe(
            df_risk.style.background_gradient(
                cmap="YlOrRd", subset=[col_sri]
            ).format({
                col_sri: "{:.3f}",
                col_gcn: "{:.3f}",
                col_bc: "{:.3f}"
            }),
            use_container_width=True,
            height=320
        )
        
        # Quick summary alert in the sidebar/right-column
        if num_critical > 0:
            st.error(texts[lang_code]["alert_state_attention"].format(num_critical=num_critical))
        else:
            st.success(texts[lang_code]["alert_state_stable"])

    # Expandable GCN Engine Mathematical Inspector
    st.markdown("---")
    with st.expander(texts[lang_code]["spectral_inspector_title"]):
        st.markdown(texts[lang_code]["spectral_formalism_title"])
        st.write(texts[lang_code]["spectral_formalism_desc"])
        
        st.latex(texts[lang_code]["spectral_formula_1"])
        st.latex(texts[lang_code]["spectral_formula_2"])
        st.latex(texts[lang_code]["spectral_formula_3"])
        st.latex(texts[lang_code]["spectral_formula_4"])
        
        st.markdown(texts[lang_code]["laplacian_visualization"])
        df_laplacian = pd.DataFrame(
            A_hat, 
            index=[node_display_names[lang_code][n].split(" (")[0] for n in node_list], 
            columns=[node_display_names[lang_code][n].split(" (")[0] for n in node_list]
        )
        st.dataframe(
            df_laplacian.style.background_gradient(cmap="viridis").format("{:.3f}"),
            use_container_width=True
        )
        st.caption(texts[lang_code]["laplacian_caption"])


# ---------------------------------------------------------
# 7. TAB 2: MITIGATION & NETWORK FLOW CONTROL
# ---------------------------------------------------------
with tab2:
    st.markdown(texts[lang_code]["mitigation_title"])
    
    if num_critical == 0:
        st.info(texts[lang_code]["mitigation_no_action"])
    else:
        # Loop through each critical node and display its detailed physics card and topological routing
        for node in critical_nodes:
            idx = node_to_idx[node]
            
            # Get raw features
            rain = X_raw[idx, 0]
            insar = X_raw[idx, 1]
            slope = X_raw[idx, 2]
            
            # Physics-Informed Geotechnical Model Parameters (Simplified Infinite Slope Model)
            # Soil Cohesion decreases exponentially with cumulative InSAR displacements (Strain softening)
            c_prime = 15.0 * np.exp(-0.08 * insar)
            # Pore water pressure increases linearly with rainfall load
            u_pore = 0.15 * rain
            # Total normal stress (assumed static)
            sigma_normal = 50.0
            # Friction angle (assumed 32 degrees for mountain soil)
            phi_deg = 32.0
            phi_rad = np.radians(phi_deg)
            slope_rad = np.radians(slope)
            
            # Infinite slope stability equation for Factor of Safety (FS)
            numerator = c_prime + (sigma_normal - u_pore) * np.tan(phi_rad)
            denominator = sigma_normal * np.tan(slope_rad)
            fs = numerator / denominator if denominator > 0 else 5.0
            
            # Render physics alert card
            alert_card_html = styles.render_physics_alert_card(
                node_name=node_display_names[lang_code][node],
                sri_val=sri[node],
                gcn_val=float(Z_gcn[idx, 0]),
                bc_val=betweenness[node],
                slope=slope,
                rain=rain,
                insar=insar,
                c_prime=c_prime,
                u_pore=u_pore,
                fs=fs,
                lang=lang_code
            )
            
            col_alert, col_route = st.columns([1, 1])
            
            with col_alert:
                st.markdown(alert_card_html, unsafe_allow_html=True)
                
            with col_route:
                st.markdown(texts[lang_code]["rerouting_analysis_title"])
                
                # Define primary energy routing pathways in the grid
                # (Generators to main regional consumption substations)
                transmissions = [
                    (texts[lang_code]["line_title_1"], "Centrale Alpina (Gen-1)", "Sottostazione Garda (Sub-3)"),
                    (texts[lang_code]["line_title_2"], "Centrale Termoelettrica (Gen-2)", "Sottostazione Val Camonica (Sub-1)"),
                    (texts[lang_code]["line_title_3"], "Centrale Alpina (Gen-1)", "Sottostazione Valtellina (Sub-2)")
                ]
                
                paths_checked = 0
                for path_title, source, target in transmissions:
                    # Find original shortest path
                    try:
                        orig_path = nx.shortest_path(G, source=source, target=target)
                    except nx.NetworkXNoPath:
                        continue
                    
                    # If the critical node is part of this path, re-routing is required
                    if node in orig_path:
                        paths_checked += 1
                        
                        # Create temporary graph without the failing node
                        G_temp = G.copy()
                        G_temp.remove_node(node)
                        
                        is_severed = False
                        rerouted_path = []
                        try:
                            rerouted_path = nx.shortest_path(G_temp, source=source, target=target)
                        except nx.NetworkXNoPath:
                            is_severed = True
                            
                        # Render the routing scenario card
                        routing_card_html = styles.render_routing_card(
                            node_name=node_display_names[lang_code][node],
                            path_name=path_title,
                            original_path=orig_path,
                            rerouted_path=rerouted_path,
                            is_severed=is_severed,
                            lang=lang_code
                        )
                        st.markdown(routing_card_html, unsafe_allow_html=True)
                
                if paths_checked == 0:
                    st.success(texts[lang_code]["no_impact_msg"])