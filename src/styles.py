import streamlit as st

def apply_base_styles():
    """
    Injects custom CSS styling and imports Google Fonts for a consistent,
    premium, engineering-focused look.
    """
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
        
        /* Main container background */
        .stApp {
            background-color: #0E1117;
            color: #ECEFF1;
            font-family: 'Outfit', sans-serif;
        }
        
        /* Header container style */
        .header-box {
            background: linear-gradient(135deg, #161A23 0%, #0E1117 100%);
            border: 1px solid #2D3748;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        }
        
        /* Titles and text styling */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            color: #FFFFFF;
        }
        
        /* Custom KPI Cards */
        .kpi-container {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }
        
        .kpi-card {
            background-color: #161A23;
            border: 1px solid #2D3748;
            border-radius: 8px;
            padding: 16px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        }
        
        .kpi-val-green {
            font-size: 2.2rem;
            font-weight: 700;
            color: #00E676;
        }
        
        .kpi-val-amber {
            font-size: 2.2rem;
            font-weight: 700;
            color: #FFB300;
        }
        
        .kpi-val-red {
            font-size: 2.2rem;
            font-weight: 700;
            color: #FF1744;
        }
        
        .kpi-label {
            font-size: 0.85rem;
            color: #90A4AE;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 4px;
        }
        
        /* Tab modifications */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #0E1117;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #161A23;
            border: 1px solid #2D3748;
            border-radius: 4px 4px 0px 0px;
            color: #ECEFF1;
            padding: 10px 20px;
            font-weight: bold;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #1E2533 !important;
            border-bottom: 3px solid #00E676 !important;
            color: #00E676 !important;
        }
        
        /* Styled expander */
        .streamlit-expanderHeader {
            background-color: #161A23 !important;
            border: 1px solid #2D3748 !important;
            border-radius: 4px !important;
            color: #FFFFFF !important;
        }
        
        .streamlit-expanderContent {
            background-color: #11141D !important;
            border: 1px solid #2D3748 !important;
            border-top: none !important;
            border-radius: 0px 0px 4px 4px !important;
        }
        
        /* Sidebar adjustments */
        section[data-testid="stSidebar"] {
            background-color: #11141C !important;
            border-right: 1px solid #2D3748;
        }
        
        /* Inline math typography */
        .katex {
            font-size: 1.05em !important;
        }
        </style>
    """, unsafe_allow_html=True)

def render_header(lang="it"):
    """
    Renders the custom header for the Terna Grid Resilience AI dashboard.
    """
    subtitle = (
        "Architettura Graph Convolutional Network (GCN) nativa per la propagazione spettrale del rischio frana idrogeologico."
        if lang == "it"
        else "Native Graph Convolutional Network (GCN) architecture for spectral propagation of hydrogeological landslide risk."
    )
    st.markdown(f"""
        <div class="header-box">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h1 style="margin: 0px; font-size: 2.2rem; letter-spacing: -0.5px;">⚡ Terna Grid Resilience AI</h1>
                    <p style="margin: 5px 0px 0px 0px; color: #90A4AE; font-size: 1.05rem;">
                        {subtitle}
                    </p>
                </div>
                <div style="text-align: right;">
                    <span style="background-color: rgba(0, 230, 118, 0.1); color: #00E676; border: 1px solid #00E676; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: bold; font-family: 'JetBrains Mono', monospace;">
                        ENGINE: NumPy + NetworkX (Pure Spectral)
                    </span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_kpis(kpi_class, kpi_status, num_critical, total_nodes, threshold, avg_sri, lang="it"):
    """
    Renders key performance indicator cards with colors matching the system state.
    """
    label_status = "STATO DEL SISTEMA" if lang == "it" else "SYSTEM STATUS"
    label_alert = f"COMPONENTI IN ALLERTA (SRI > {threshold})" if lang == "it" else f"COMPONENTS IN ALERT (SRI > {threshold})"
    label_avg = "INDICE DI RISCHIO SISTEMICO MEDIO (SRI)" if lang == "it" else "AVERAGE SYSTEMIC RISK INDEX (SRI)"
    
    st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-card">
                <div class="{kpi_class}">{kpi_status}</div>
                <div class="kpi-label">{label_status}</div>
            </div>
            <div class="kpi-card">
                <div class="{kpi_class}">{num_critical} / {total_nodes}</div>
                <div class="kpi-label">{label_alert}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-val-amber">{avg_sri:.3f}</div>
                <div class="kpi-label">{label_avg}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_physics_alert_card(node_name, sri_val, gcn_val, bc_val, slope, rain, insar, c_prime, u_pore, fs, lang="it"):
    """
    Renders an HTML card representing the geotechnical slope stability status
    and physical constraints for a node in alert.
    Cleaned from LaTeX formulas to prevent Streamlit parsing errors.
    """
    if fs < 1.0:
        fs_color = "#FF1744"
        fs_status = "INSTABILE (FS < 1.0)" if lang == "it" else "UNSTABLE (FS < 1.0)"
    elif fs < 1.3:
        fs_color = "#FFB300"
        fs_status = "CRITICO (1.0 ≤ FS < 1.3)" if lang == "it" else "CRITICAL (1.0 ≤ FS < 1.3)"
    else:
        fs_color = "#00E676"
        fs_status = "STABILE (FS ≥ 1.3)" if lang == "it" else "STABLE (FS ≥ 1.3)"
        
    title = "⚠️ DISSESTO IDROGEOLOGICO RILEVATO" if lang == "it" else "⚠️ HYDROGEOLOGICAL INSTABILITY DETECTED"
    
    if lang == "it":
        desc = (
            f"<b>Analisi Geotecnica del Terreno (Infinite Slope Stability):</b><br>"
            f"Le piogge accumulate di <b>{rain:.1f} mm</b> innalzano la pressione interstiziale a <b>u = {u_pore:.2f} kPa</b>.<br>"
            f"I micro-spostamenti del versante registrati via satellite a <b>{insar:.2f} mm/mese</b> causano lo snervamento delle argille terrestri, "
            f"riducendo la coesione del terreno a <b>C' = {c_prime:.2f} kPa</b>. Sulla pendenza di <b>{slope:.1f}°</b>, l'interazione "
            f"R_v x I_v riduce il coefficiente di stabilità meccanica."
        )
    else:
        desc = (
            f"<b>Geotechnical Soil Analysis (Infinite Slope Stability):</b><br>"
            f"Cumulative rainfall of <b>{rain:.1f} mm</b> increases pore water pressure to <b>u = {u_pore:.2f} kPa</b>.<br>"
            f"Satellite-recorded slope micro-displacements of <b>{insar:.2f} mm/month</b> cause shear softening of clay soils, "
            f"reducing soil cohesion to <b>C' = {c_prime:.2f} kPa</b>. On a slope of <b>{slope:.1f}°</b>, the interaction "
            f"R_v x I_v reduces the mechanical stability coefficient."
        )
        
    lbl_topology = "Topologia (BC)" if lang == "it" else "Topology (BC)"
    lbl_stability = "Fattore Stabilità" if lang == "it" else "Stability Factor"
    lbl_physical_state = "Stato Strutturale Fisico" if lang == "it" else "Physical Structural State"

    # Linearized HTML string without multi-line indentations to ensure native browser parsing
    card_html = f"""<div style="background-color: #161A23; border: 1px solid #FF1744; border-radius: 8px; padding: 18px; margin-bottom: 16px; box-shadow: 0 4px 12px rgba(255, 23, 68, 0.15);"><div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #2D3748; padding-bottom: 8px; margin-bottom: 12px;"><span style="font-size: 0.95rem; font-weight: bold; color: #FF1744; font-family: 'Outfit', sans-serif;">{title}</span><span style="background-color: rgba(255, 23, 68, 0.15); color: #FF1744; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: bold; font-family: 'JetBrains Mono', monospace;">SRI: {sri_val:.3f}</span></div><h4 style="margin: 0 0 10px 0; color: #FFFFFF; font-size: 1.15rem; font-family: 'Outfit', sans-serif;">{node_name}</h4><p style="font-size: 0.9rem; color: #ECEFF1; margin: 0 0 14px 0; line-height: 1.45; font-family: 'Outfit', sans-serif;">{desc}</p><div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; background-color: #0E1117; padding: 10px; border-radius: 6px; text-align: center; border: 1px solid #2D3748;"><div><div style="font-size: 0.75rem; color: #90A4AE; font-family: 'Outfit', sans-serif; text-transform: uppercase;">GCN Vuln (Z)</div><div style="font-size: 1rem; font-weight: 700; color: #FFB300; font-family: 'JetBrains Mono', monospace; margin-top: 2px;">{gcn_val:.3f}</div></div><div><div style="font-size: 0.75rem; color: #90A4AE; font-family: 'Outfit', sans-serif; text-transform: uppercase;">{lbl_topology}</div><div style="font-size: 1rem; font-weight: 700; color: #00E676; font-family: 'JetBrains Mono', monospace; margin-top: 2px;">{bc_val:.3f}</div></div><div><div style="font-size: 0.75rem; color: #90A4AE; font-family: 'Outfit', sans-serif; text-transform: uppercase;">{lbl_stability}</div><div style="font-size: 1rem; font-weight: 700; color: {fs_color}; font-family: 'JetBrains Mono', monospace; margin-top: 2px;">FS = {fs:.2f}</div></div></div><div style="margin-top: 12px; font-size: 0.85rem; color: {fs_color}; font-weight: bold; font-family: 'Outfit', sans-serif;">{lbl_physical_state}: {fs_status}</div></div>"""
    return card_html

def render_routing_card(node_name, path_name, original_path, rerouted_path, is_severed, lang="it"):
    """
    Renders an HTML card representing the energy re-routing state for a grid link.
    """
    if is_severed:
        title = "💥 INTERRUZIONE DI DIRETTRICE" if lang == "it" else "💥 PATHWAY DISRUPTION"
        status_badge = "SEVERED"
        desc = (
            f"La disattivazione del nodo <b>{node_name}</b> isola elettricamente la destinazione. Non vi sono percorsi alternativi."
            if lang == "it"
            else f"Deactivation of node <b>{node_name}</b> electrically isolates the destination. There are no alternative paths."
        )
        msg_critical = (
            "[CRITICAL] Grid partitioned. Avviare generatori d'emergenza locali."
            if lang == "it"
            else "[CRITICAL] Grid partitioned. Start local emergency generators."
        )
        card_html = f"""
        <div style="
            background-color: #161A23;
            border: 1px solid #FF1744;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
            box-shadow: 0 4px 8px rgba(255, 23, 68, 0.08);
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #2D3748; padding-bottom: 8px; margin-bottom: 10px;">
                <span style="font-size: 0.9rem; font-weight: bold; color: #FF1744; font-family: 'Outfit', sans-serif;">{title}</span>
                <span style="background-color: rgba(255, 23, 68, 0.2); color: #FF1744; padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; font-weight: bold; font-family: 'JetBrains Mono', monospace;">{status_badge}</span>
            </div>
            <h4 style="margin: 0 0 6px 0; color: #FFFFFF; font-size: 1.05rem; font-family: 'Outfit', sans-serif;">{path_name}</h4>
            <p style="font-size: 0.85rem; color: #B0BEC5; margin: 0 0 10px 0; line-height: 1.4; font-family: 'Outfit', sans-serif;">
                {desc}
            </p>
            <div style="background-color: rgba(255, 23, 68, 0.08); border: 1px solid #FF1744; color: #FF1744; padding: 8px; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;">
                {msg_critical}
            </div>
        </div>
        """
    else:
        orig_str = " ➔ ".join([p.split(" (")[0] for p in original_path])
        reroute_str = " ➔ ".join([p.split(" (")[0] for p in rerouted_path])
        
        title = "🛡️ ADJACENCY RE-ROUTING COMPILATO" if lang == "it" else "🛡️ ADJACENCY RE-ROUTING COMPLETED"
        status_badge = "RE-ROUTED"
        desc = (
            f"Flusso re-instradato con successo aggirando il bottleneck <b>{node_name}</b>."
            if lang == "it"
            else f"Flow successfully rerouted bypassing the bottleneck <b>{node_name}</b>."
        )
        lbl_orig = "Percorso Originario" if lang == "it" else "Original Path"
        lbl_reroute = "Percorso Re-routed" if lang == "it" else "Rerouted Path"
        
        card_html = f"""
        <div style="
            background-color: #161A23;
            border: 1px solid #00E676;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
            box-shadow: 0 4px 8px rgba(0, 230, 118, 0.08);
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #2D3748; padding-bottom: 8px; margin-bottom: 10px;">
                <span style="font-size: 0.9rem; font-weight: bold; color: #00E676; font-family: 'Outfit', sans-serif;">{title}</span>
                <span style="background-color: rgba(0, 230, 118, 0.2); color: #00E676; padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; font-weight: bold; font-family: 'JetBrains Mono', monospace;">{status_badge}</span>
            </div>
            <h4 style="margin: 0 0 6px 0; color: #FFFFFF; font-size: 1.05rem; font-family: 'Outfit', sans-serif;">{path_name}</h4>
            <p style="font-size: 0.85rem; color: #B0BEC5; margin: 0 0 10px 0; line-height: 1.4; font-family: 'Outfit', sans-serif;">
                {desc}
            </p>
            <div style="font-size: 0.8rem; color: #ECEFF1; font-family: 'JetBrains Mono', monospace; line-height: 1.5;">
                <div style="margin-bottom: 4px;"><span style="color: #78909C; font-weight: bold;">{lbl_orig}:</span> <span style="text-decoration: line-through; color: #78909C;">{orig_str}</span></div>
                <div><span style="color: #00E676; font-weight: bold;">{lbl_reroute}:</span> <span style="color: #00E676; font-weight: bold;">{reroute_str}</span></div>
            </div>
        </div>
        """
    return card_html
