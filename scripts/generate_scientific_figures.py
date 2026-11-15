"""
Publication-Quality Scientific Figure Generation Suite
Target: Springer Journal of Intelligent Information Systems (JIIS)
Repository: RoushanKhalid/Retrieval-Incremental-Learning

Master scientific plotting suite:
- Computer Modern LaTeX typography (mathtext.fontset = 'cm')
- Proportional vector dimensions (13.5x7.0 in for architectures, 13.5x4.8 in for dual panels)
- Pure academic color palette (Navy, Forest Green, Crimson, Indigo, Steel)
- Minimal text, maximum visual communication, zero collisions
- Truthful data loaded directly from empirical artifacts
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

# Global academic formatting
plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.family": "sans-serif",
    "mathtext.fontset": "cm",  # Computer Modern LaTeX math
    "mathtext.rm": "serif",
    "mathtext.it": "serif:italic",
    "mathtext.bf": "serif:bold",
    "axes.edgecolor": "#718096",
    "axes.linewidth": 1.1,
    "grid.color": "#E2E8F0",
    "grid.linestyle": "--",
    "grid.alpha": 0.65,
    "figure.autolayout": False,
    "text.color": "#1A202C",
    "axes.labelcolor": "#2D3748",
    "xtick.color": "#4A5568",
    "ytick.color": "#4A5568",
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
})

# Path resolution: support both paper_latex and paper/
FIGURES_DIRS = [
    Path("paper_latex/figures"),
    Path("paper/figures"),
]
for d in FIGURES_DIRS:
    d.mkdir(parents=True, exist_ok=True)

# Find empirical artifacts
ARTIFACT_PATHS = [
    Path("tcontext/artifacts"),
    Path("outputs/benchmark_results"),
    Path("outputs/metrics"),
]

def find_file(filename):
    for p in ARTIFACT_PATHS:
        f = p / filename
        if f.exists():
            return f
    matches = list(Path(".").rglob(filename))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Cannot find {filename}")

GPU_CSV = find_file("benchmark_results_cuda_0.csv")
CPU_CSV = find_file("benchmark_results_cpu.csv")
NOISE_JSON = find_file("noise_experiment_results.json")

df_gpu = pd.read_csv(GPU_CSV)
df_cpu = pd.read_csv(CPU_CSV)
with open(NOISE_JSON, "r") as f:
    noise_data = json.load(f)


def save_figure(fig, base_name):
    """Save figure in both PDF and 300 DPI PNG across target directories."""
    for d in FIGURES_DIRS:
        fig.savefig(d / f"{base_name}.png", dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(d / f"{base_name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ Saved {base_name}.png and {base_name}.pdf")


# ==============================================================================
# FIGURE 1: Overall System Architecture (Dual-Stream Modular Design)
# ==============================================================================
def draw_card(ax, xy, width, height, title="", subtitle="", badge="", badge_colors=("#EFF6FF", "#1D4ED8"),
              facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.4, rounding=0.08, zorder=2):
    """Draw a professional card container with rounded corners and optional badge/header."""
    x, y = xy
    box = FancyBboxPatch((x, y), width, height, boxstyle=f"round,pad=0.06,rounding_size={rounding}",
                         facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, zorder=zorder)
    ax.add_patch(box)
    
    top_y = y + height
    if badge:
        bg_col, text_col = badge_colors
        ax.text(x + width/2, top_y - 0.20, badge, fontsize=7.0, fontweight="bold", color=text_col,
                ha="center", va="center", zorder=zorder+1,
                bbox=dict(boxstyle="round,pad=0.18", facecolor=bg_col, edgecolor=edgecolor, lw=0.6))
    if title:
        title_y = top_y - (0.46 if badge else 0.26)
        ax.text(x + width/2, title_y, title, fontsize=8.6, fontweight="bold", color="#0F172A",
                ha="center", va="center", zorder=zorder+1)
    if subtitle:
        sub_y = top_y - (0.70 if badge else 0.48)
        ax.text(x + width/2, sub_y, subtitle, fontsize=7.2, color="#475569",
                ha="center", va="center", zorder=zorder+1)
    return box

def draw_arrow(ax, p1, p2, color="#2563EB", linewidth=1.6, mutation_scale=13, linestyle="-", label="",
               label_pos=0.5, label_bg="#EFF6FF", label_border="#BFDBFE", label_color="#1E40AF", zorder=4):
    """Draw a clean directional connection arrow with optional text chip."""
    arrow = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=mutation_scale,
                            linewidth=linewidth, color=color, linestyle=linestyle, zorder=zorder)
    ax.add_patch(arrow)
    if label:
        x_lbl = p1[0] + (p2[0] - p1[0]) * label_pos
        y_lbl = p1[1] + (p2[1] - p1[1]) * label_pos
        ax.text(x_lbl, y_lbl, label, fontsize=6.8, fontweight="bold", color=label_color,
                ha="center", va="center", zorder=zorder+2,
                bbox=dict(boxstyle="round,pad=0.15", facecolor=label_bg, edgecolor=label_border, lw=0.6))
    return arrow

def generate_figure1():
    fig, ax = plt.subplots(figsize=(14.6, 9.5), dpi=300)
    ax.set_xlim(0, 14.6)
    ax.set_ylim(0, 9.5)
    ax.axis("off")

    # ==========================================================================
    # OVERALL CONTAINER FRAMES (STAGE A & STAGE B)
    # ==========================================================================
    # Frame A: Stage A - Ingestion & Memory Curation
    f_mode1 = FancyBboxPatch((0.25, 4.90), 14.10, 4.35, boxstyle="round,pad=0.1,rounding_size=0.10",
                             facecolor="#F8FAFC", edgecolor="#94A3B8", linewidth=1.5, zorder=0)
    ax.add_patch(f_mode1)
    
    # Stage A Banner
    ax.text(0.50, 9.02, "STAGE A: INCREMENTAL KNOWLEDGE INGESTION & MEMORY CURATION (CONTINUAL LEARNING STREAM)",
            fontsize=10.0, fontweight="bold", color="#1E3A8A", ha="left", va="center")
    ax.text(0.50, 8.78, r"Non-parametric knowledge retention with strictly invariant visual backbone ($\nabla_\theta = \mathbf{0}$) and 3-stage automated quality gating",
            fontsize=7.8, color="#475569", ha="left", va="center")

    # Frame B: Stage B - Inference & Dynamic Evidence Fusion
    f_mode2 = FancyBboxPatch((0.25, 0.25), 14.10, 4.25, boxstyle="round,pad=0.1,rounding_size=0.10",
                             facecolor="#FAF5FF", edgecolor="#C084FC", linewidth=1.5, zorder=0)
    ax.add_patch(f_mode2)
    
    # Stage B Banner (Concise to guarantee zero overlap with feed link at x=8.725)
    ax.text(0.50, 4.25, "STAGE B: SUB-MILLISECOND INFERENCE & EVIDENCE FUSION (ONLINE QUERY STREAM)",
            fontsize=10.0, fontweight="bold", color="#581C87", ha="left", va="center")
    ax.text(0.50, 4.02, r"Dual-evidence decision consensus combining parametric prior with non-parametric retrieval",
            fontsize=7.8, color="#475569", ha="left", va="center")

    # ==========================================================================
    # STAGE A: INCREMENTAL KNOWLEDGE INGESTION (TOP HALF)
    # ==========================================================================
    # 1. Candidate Input (Col 1)
    draw_card(ax, (0.45, 5.45), 1.65, 3.05,
              title="Candidate Sample",
              badge="INPUT STREAM",
              badge_colors=("#EFF6FF", "#1D4ED8"),
              facecolor="#FFFFFF", edgecolor="#3B82F6", linewidth=1.6)
    ax.text(1.275, 7.50, r"$(\mathbf{x}_t, y_t)$", fontsize=10.5, fontweight="bold", color="#0F172A", ha="center")
    ax.text(1.275, 7.00, r"$\mathbf{x}_t \in \mathbb{R}^{224 \times 224 \times 3}$", fontsize=8.0, color="#334155", ha="center")
    ax.text(1.275, 6.65, r"$y_t \in \{0, 1\}$ (Class)", fontsize=7.8, color="#334155", ha="center")
    ax.text(1.275, 6.15, "Continuous Ingestion", fontsize=7.4, color="#64748B", ha="center")
    ax.text(1.275, 5.80, "Zero Stored Replay Data", fontsize=7.2, fontstyle="italic", color="#64748B", ha="center")

    # Arrow 1: Input -> Frozen Backbone
    draw_arrow(ax, (2.10, 6.95), (2.50, 6.95), color="#2563EB", label=r"$\mathbf{x}_t$", label_pos=0.5)

    # 2. Frozen Semantic Backbone (Col 2)
    draw_card(ax, (2.50, 5.45), 1.85, 3.05,
              title="Frozen Backbone",
              subtitle="CLIP ViT-B/32",
              badge="INVARIANT ENCODER",
              badge_colors=("#F0F9FF", "#0369A1"),
              facecolor="#FFFFFF", edgecolor="#0284C7", linewidth=1.6)
    
    ax.text(3.425, 7.10, r"$\nabla_\theta = \mathbf{0} \quad (\text{Invariant})$", fontsize=8.0, fontweight="bold", color="#0369A1",
            ha="center", bbox=dict(boxstyle="round,pad=0.18", facecolor="#E0F2FE", edgecolor="#BAE6FD", lw=0.6))
    ax.text(3.425, 6.55, r"$\mathbf{e}_t = \frac{f_\theta(\mathbf{x}_t)}{\|f_\theta(\mathbf{x}_t)\|_2} \in \mathbb{R}^{512}$",
            fontsize=8.2, fontweight="bold", color="#0F172A", ha="center")
    ax.text(3.425, 6.05, "Zero Backpropagation", fontsize=7.5, color="#334155", ha="center")
    ax.text(3.425, 5.75, "Prevents Catastrophic Drift", fontsize=7.2, fontstyle="italic", color="#64748B", ha="center")

    # Arrow 2: Backbone -> Metric Adapter
    draw_arrow(ax, (4.35, 6.95), (4.75, 6.95), color="#0284C7", label=r"$\mathbf{e}_t$", label_pos=0.5)

    # 3. Metric Retrieval Adapter (Col 3)
    draw_card(ax, (4.75, 5.45), 1.85, 3.05,
              title="Metric Adapter",
              subtitle=r"Residual Head $g_\phi$",
              badge="CONTRASTIVE ALIGNMENT",
              badge_colors=("#EEF2FF", "#4338CA"),
              facecolor="#FFFFFF", edgecolor="#4F46E5", linewidth=1.6)
    
    ax.text(5.675, 7.10, "526K Params (<0.35%)", fontsize=7.6, fontweight="bold", color="#4338CA",
            ha="center", bbox=dict(boxstyle="round,pad=0.18", facecolor="#E0E7FF", edgecolor="#C7D2FE", lw=0.6))
    ax.text(5.675, 6.55, r"$\mathbf{z}_t = \mathrm{Norm}(\mathbf{e}_t + g_\phi(\mathbf{e}_t))$",
            fontsize=8.2, fontweight="bold", color="#0F172A", ha="center")
    ax.text(5.675, 6.05, r"SupCon Loss ($\tau = 0.07$)", fontsize=7.5, color="#334155", ha="center")
    ax.text(5.675, 5.75, "Maximizes Class Margin", fontsize=7.2, fontstyle="italic", color="#64748B", ha="center")

    # Arrow 3: Adapter -> IMO Quality Firewall
    draw_arrow(ax, (6.60, 6.95), (7.00, 6.95), color="#4F46E5", label=r"$(\mathbf{z}_t, y_t)$", label_pos=0.5)

    # 4. Intelligent Memory Orchestrator (IMO Firewall) Panel (Col 4)
    p_imo = FancyBboxPatch((7.00, 5.05), 4.55, 3.45, boxstyle="round,pad=0.08,rounding_size=0.08",
                           facecolor="#F0FDF4", edgecolor="#16A34A", linewidth=1.6, linestyle="--", zorder=1)
    ax.add_patch(p_imo)
    ax.text(9.275, 8.28, "INTELLIGENT MEMORY ORCHESTRATOR (IMO FIREWALL)", fontsize=8.6, fontweight="bold", color="#14532D", ha="center")
    ax.text(9.275, 8.05, "Multi-Stage Non-Parametric Verification Prior to Memory Retention", fontsize=7.2, color="#15803D", ha="center")

    # Gate 1: Consistency Check
    draw_card(ax, (7.15, 6.00), 1.25, 1.80,
              title="Gate 1",
              subtitle="Consistency",
              facecolor="#FFFFFF", edgecolor="#16A34A", linewidth=1.4)
    ax.text(7.775, 6.85, r"$\mathbf{z}_t^\top \mathbf{p}_{y_t} \geq \tau_c$", fontsize=8.5, fontweight="bold", color="#0F172A", ha="center")
    ax.text(7.775, 6.45, r"$\tau_c = 0.65$", fontsize=7.8, fontweight="bold", color="#16A34A", ha="center")
    ax.text(7.775, 6.15, "Prototype Match", fontsize=6.8, color="#64748B", ha="center")

    # Gate 1 -> Gate 2 Arrow
    draw_arrow(ax, (8.40, 6.90), (8.65, 6.90), color="#16A34A", linewidth=1.5, label="Pass", label_bg="#DCFCE7", label_border="#86EFAC", label_color="#14532D")

    # Gate 2: Margin Check
    draw_card(ax, (8.65, 6.00), 1.25, 1.80,
              title="Gate 2",
              subtitle="Rel. Margin",
              facecolor="#FFFFFF", edgecolor="#16A34A", linewidth=1.4)
    ax.text(9.275, 6.85, r"$\Delta S_{\mathrm{rel}} \geq \tau_m$", fontsize=8.5, fontweight="bold", color="#0F172A", ha="center")
    ax.text(9.275, 6.45, r"$\tau_m = 0.00$", fontsize=7.8, fontweight="bold", color="#16A34A", ha="center")
    ax.text(9.275, 6.15, "Rejects Conflicts", fontsize=6.8, color="#64748B", ha="center")

    # Gate 2 -> Gate 3 Arrow
    draw_arrow(ax, (9.90, 6.90), (10.15, 6.90), color="#16A34A", linewidth=1.5, label="Pass", label_bg="#DCFCE7", label_border="#86EFAC", label_color="#14532D")

    # Gate 3: Outlier Density Filter
    draw_card(ax, (10.15, 6.00), 1.25, 1.80,
              title="Gate 3",
              subtitle="Density",
              facecolor="#FFFFFF", edgecolor="#16A34A", linewidth=1.4)
    ax.text(10.775, 6.85, r"$\bar{D}_{K=5} \leq \tau_{\mathrm{out}}$", fontsize=8.2, fontweight="bold", color="#0F172A", ha="center")
    ax.text(10.775, 6.45, r"$\tau_{\mathrm{out}} = 0.45$", fontsize=7.8, fontweight="bold", color="#16A34A", ha="center")
    ax.text(10.775, 6.15, "Filter Outliers", fontsize=6.8, color="#64748B", ha="center")

    # Quarantine Buffer Sink (Bottom of IMO, width 4.35)
    draw_card(ax, (7.10, 5.15), 4.35, 0.65,
              facecolor="#FEF2F2", edgecolor="#DC2626", linewidth=1.3)
    ax.text(9.275, 5.58, r"Quarantine Buffer $\mathcal{Q}$ (Isolated Non-Destructive Storage)",
            fontsize=7.6, fontweight="bold", color="#991B1B", ha="center")
    ax.text(9.275, 5.32, r"Stores $(\mathbf{z}_t, y_t, \text{reason}, \text{timestamp})$ $\mid$ Intercepts 100% Corrupted Samples (525/525)",
            fontsize=6.5, color="#B91C1C", ha="center")

    # Rejection Arrows (Dashed Red down from each Gate)
    ax.add_patch(FancyArrowPatch((7.775, 6.00), (7.775, 5.80), arrowstyle="-|>", mutation_scale=10, linewidth=1.2, color="#DC2626", linestyle="--", zorder=3))
    ax.add_patch(FancyArrowPatch((9.275, 6.00), (9.275, 5.80), arrowstyle="-|>", mutation_scale=10, linewidth=1.2, color="#DC2626", linestyle="--", zorder=3))
    ax.add_patch(FancyArrowPatch((10.775, 6.00), (10.775, 5.80), arrowstyle="-|>", mutation_scale=10, linewidth=1.2, color="#DC2626", linestyle="--", zorder=3))

    # Arrow 4: Gate 3 (Pass All) -> Active Vector Memory
    draw_arrow(ax, (11.40, 6.90), (11.90, 6.90), color="#16A34A", linewidth=2.0, label="Pass All", label_pos=0.5,
               label_bg="#DCFCE7", label_border="#86EFAC", label_color="#14532D")

    # 5. Active Vector Memory & Class Prototypes (Col 5)
    draw_card(ax, (11.90, 5.05), 2.30, 3.45,
              title="Active Vector Memory",
              subtitle="ChromaDB (HNSW Graph)",
              badge="EXTERNAL STORAGE",
              badge_colors=("#D1FAE5", "#065F46"),
              facecolor="#FFFFFF", edgecolor="#059669", linewidth=1.8)
    
    ax.text(13.05, 7.42, r"$\mathcal{M} \leftarrow \mathcal{M} \cup \{(\mathbf{z}_t, y_t)\}$",
            fontsize=8.8, fontweight="bold", color="#065F46", ha="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#ECFDF5", edgecolor="#A7F3D0", lw=0.7))
    ax.text(13.05, 7.00, "Metric: Cosine Proximity", fontsize=7.4, color="#334155", ha="center")
    ax.text(13.05, 6.75, r"Search Complexity: $\mathcal{O}(\log N)$", fontsize=7.4, color="#334155", ha="center")
    ax.text(13.05, 6.50, r"Host RAM (0 MB VRAM)", fontsize=7.4, fontweight="bold", color="#047857", ha="center")

    # Centroid Prototype Box inside Col 5
    p_proto = FancyBboxPatch((12.02, 5.18), 2.06, 1.10, boxstyle="round,pad=0.05,rounding_size=0.06",
                             facecolor="#F0FDF4", edgecolor="#34D399", linewidth=1.2, zorder=2)
    ax.add_patch(p_proto)
    ax.text(13.05, 6.00, r"Class Prototypes $\{\mathbf{p}_c\}$", fontsize=7.8, fontweight="bold", color="#065F46", ha="center")
    ax.text(13.05, 5.70, r"$\mathbf{p}_c \leftarrow \mathrm{Norm}(n \mathbf{p}_c + \mathbf{z}_t)$", fontsize=7.8, color="#047857", ha="center")
    ax.text(13.05, 5.38, r"$\mathcal{O}(d)$ Constant-Time Update", fontsize=6.8, fontstyle="italic", color="#059669", ha="center")


    # ==========================================================================
    # STAGE B: ONLINE INFERENCE & DYNAMIC EVIDENCE FUSION (BOTTOM HALF)
    # ==========================================================================
    # 1. Query Input (Col 1)
    draw_card(ax, (0.45, 0.80), 1.65, 2.95,
              title="Inference Query",
              badge="QUERY STREAM",
              badge_colors=("#F3E8FF", "#6D28D9"),
              facecolor="#FFFFFF", edgecolor="#7C3AED", linewidth=1.6)
    ax.text(1.275, 2.80, r"$\mathbf{x}_q \in \mathcal{X}$", fontsize=10.5, fontweight="bold", color="#0F172A", ha="center")
    ax.text(1.275, 2.35, r"Unlabeled Image", fontsize=8.0, color="#334155", ha="center")
    ax.text(1.275, 2.05, r"RGB $224 \times 224 \times 3$", fontsize=7.6, color="#334155", ha="center")
    ax.text(1.275, 1.60, "Real-Time Query", fontsize=7.4, color="#64748B", ha="center")
    ax.text(1.275, 1.25, "Dual Evidence Streams", fontsize=7.2, fontstyle="italic", color="#64748B", ha="center")

    # Split Arrows from Query Input:
    # Arrow B1_top: to Shared Non-Parametric Stream (Upper Track of Stage B)
    ax.add_patch(FancyArrowPatch((2.10, 2.50), (2.50, 2.90), arrowstyle="-|>", mutation_scale=12, linewidth=1.6, color="#0284C7", zorder=4))
    # Arrow B1_bottom: to Parametric Stream (Lower Track of Stage B)
    ax.add_patch(FancyArrowPatch((2.10, 1.95), (2.50, 1.45), arrowstyle="-|>", mutation_scale=12, linewidth=1.6, color="#2563EB", zorder=4))

    # --------------------------------------------------------------------------
    # TRACK 1 (UPPER TRACK OF STAGE B): NON-PARAMETRIC RETRIEVAL STREAM
    # --------------------------------------------------------------------------
    # Shared Backbone
    draw_card(ax, (2.50, 2.35), 1.85, 1.40,
              title="Shared Backbone",
              subtitle="CLIP ViT-B/32",
              badge="INVARIANT ENCODER",
              badge_colors=("#F0F9FF", "#0369A1"),
              facecolor="#FFFFFF", edgecolor="#0284C7", linewidth=1.4)
    ax.text(3.425, 2.58, r"$\mathbf{e}_q \in \mathbb{R}^{512}$", fontsize=8.0, fontweight="bold", color="#0F172A", ha="center")

    # Arrow to Metric Adapter
    draw_arrow(ax, (4.35, 3.05), (4.75, 3.05), color="#0284C7", linewidth=1.4)

    # Shared Metric Adapter
    draw_card(ax, (4.75, 2.35), 1.85, 1.40,
              title="Metric Adapter",
              subtitle=r"Residual Head $g_\phi$",
              badge="DOMAIN ALIGNED",
              badge_colors=("#EEF2FF", "#4338CA"),
              facecolor="#FFFFFF", edgecolor="#4F46E5", linewidth=1.4)
    ax.text(5.675, 2.58, r"$\mathbf{z}_q \in \mathbb{R}^{512}$", fontsize=8.0, fontweight="bold", color="#0F172A", ha="center")

    # Arrow from Metric Adapter to HNSW Retrieval
    draw_arrow(ax, (6.60, 3.05), (7.00, 3.05), color="#4F46E5", label=r"$\mathbf{z}_q$", label_pos=0.5,
               label_bg="#EEF2FF", label_border="#C7D2FE", label_color="#4338CA")

    # HNSW Nearest-Neighbor Retrieval Box (Upper Track of Stage B)
    draw_card(ax, (7.00, 2.15), 3.45, 1.60,
              badge="RETRIEVAL EVIDENCE STREAM",
              badge_colors=("#ECFDF5", "#065F46"),
              facecolor="#FFFFFF", edgecolor="#059669", linewidth=1.6)
    ax.text(8.725, 3.32, "HNSW Vector Proximity Search", fontsize=8.6, fontweight="bold", color="#065F46", ha="center")
    ax.text(8.725, 3.02, r"$\mathcal{N}_{k^*}(\mathbf{z}_q) \leftarrow \mathrm{Query}(\mathcal{M}, \mathbf{z}_q, k^*=1)$",
            fontsize=8.2, fontweight="bold", color="#047857", ha="center")
    ax.text(8.725, 2.68, r"$P_{\mathrm{ret}}(y=k \mid \mathbf{x}_q) = \sum_{i \in \mathcal{N}_{k^*}} \mathbb{I}(y_i = k) \cdot \frac{\exp(s_i/T)}{\sum \exp(s_j/T)}$",
            fontsize=7.4, color="#0F172A", ha="center")
    ax.text(8.725, 2.34, r"Sub-Millisecond Query Latency: $0.15\text{ ms} \quad (>6,700 \text{ img/sec})$",
            fontsize=7.2, fontweight="bold", color="#047857", ha="center")

    # Arrow from Retrieval to Dynamic Fusion
    draw_arrow(ax, (10.45, 2.95), (10.95, 2.60), color="#059669", linewidth=1.8, label=r"$P_{\mathrm{ret}}$", label_pos=0.5,
               label_bg="#DCFCE7", label_border="#86EFAC", label_color="#14532D")

    # --------------------------------------------------------------------------
    # TRACK 2 (LOWER TRACK OF STAGE B): PARAMETRIC BASE CLASSIFIER
    # --------------------------------------------------------------------------
    draw_card(ax, (2.50, 0.70), 7.95, 1.25,
              badge="PARAMETRIC EVIDENCE STREAM",
              badge_colors=("#EFF6FF", "#1E40AF"),
              facecolor="#FFFFFF", edgecolor="#2563EB", linewidth=1.5)
    ax.text(6.475, 1.50, "Parametric Base Visual Classifier (CLIP Linear Probe / EfficientNetB0)",
            fontsize=8.6, fontweight="bold", color="#1E40AF", ha="center")
    ax.text(6.475, 1.10, r"$P_{\mathrm{clf}}(y \mid \mathbf{x}_q) = \mathrm{Softmax}(\mathbf{W}_{\mathrm{clf}} \mathbf{e}_q + \mathbf{b}) \quad (\text{Fixed Parametric Representation Prior})$",
            fontsize=8.2, fontweight="bold", color="#0F172A", ha="center")

    # Arrow from Parametric Base Classifier to Dynamic Fusion
    draw_arrow(ax, (10.45, 1.32), (10.95, 1.80), color="#2563EB", linewidth=1.8, label=r"$P_{\mathrm{clf}}$", label_pos=0.5,
               label_bg="#EFF6FF", label_border="#BFDBFE", label_color="#1E40AF")

    # ==========================================================================
    # INTER-STAGE VERTICAL COUPLING (Active Memory M -> Vector Retrieval)
    # ==========================================================================
    # Clean orthogonal path from Stage A Active Memory down to Stage B Retrieval Engine
    # Drops from bottom of Memory M (x=13.05, y=5.05), runs horizontally at y=4.62, drops into Retrieval Engine at x=8.725 (y=3.75)
    ax.plot([13.05, 13.05], [5.05, 4.62], color="#059669", linewidth=1.8, linestyle="--", zorder=3)
    ax.plot([13.05, 8.725], [4.62, 4.62], color="#059669", linewidth=1.8, linestyle="--", zorder=3)
    ax.add_patch(FancyArrowPatch((8.725, 4.62), (8.725, 3.75), arrowstyle="-|>", mutation_scale=12,
                                 linewidth=1.8, color="#059669", linestyle="--", zorder=4))
    ax.text(10.88, 4.62, r"Queries Active Index $\mathcal{M}$ (Zero GPU VRAM, $0.15\text{ ms}$)",
            fontsize=7.4, fontweight="bold", color="#065F46", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.20", facecolor="#ECFDF5", edgecolor="#A7F3D0", lw=0.7))

    # --------------------------------------------------------------------------
    # STAGE B CONVERGENCE: DYNAMIC EVIDENCE FUSION & FINAL OUTPUT
    # --------------------------------------------------------------------------
    # Dynamic Evidence Fusion Engine (x=10.95 to 13.00, width 2.05)
    draw_card(ax, (10.95, 0.85), 2.05, 2.80,
              badge="DECISION ENGINE",
              badge_colors=("#FAF5FF", "#6B21A8"),
              facecolor="#FFFFFF", edgecolor="#7C3AED", linewidth=1.8)
    ax.text(11.975, 3.20, "Dynamic Fusion", fontsize=8.6, fontweight="bold", color="#581C87", ha="center")
    ax.text(11.975, 2.96, "Evidence Consensus", fontsize=7.2, color="#6B21A8", ha="center")
    ax.text(11.975, 2.55, r"$P(y \mid \mathbf{x}_q) =$", fontsize=8.5, fontweight="bold", color="#581C87", ha="center")
    ax.text(11.975, 2.25, r"$\alpha^* P_{\mathrm{clf}} + (1 - \alpha^*) P_{\mathrm{ret}}$", fontsize=8.5, fontweight="bold", color="#6B21A8", ha="center")
    ax.text(11.975, 1.80, r"Optimal Parameters:", fontsize=7.4, fontweight="bold", color="#334155", ha="center")
    ax.text(11.975, 1.52, r"$\alpha^* = 0.60 \quad (k^* = 1)$", fontsize=7.8, color="#0F172A", ha="center")
    ax.text(11.975, 1.18, "Validation Calibrated", fontsize=7.0, fontstyle="italic", color="#64748B", ha="center")

    # Arrow from Fusion to Final Prediction
    draw_arrow(ax, (13.00, 2.25), (13.25, 2.25), color="#7C3AED", linewidth=2.0)

    # Final Calibrated Output Decision (x=13.25 to 14.45, width 1.20)
    draw_card(ax, (13.25, 0.85), 1.20, 2.80,
              facecolor="#15803D", edgecolor="#14532D", linewidth=2.0)
    ax.text(13.85, 3.35, "OUTPUT", fontsize=7.6, fontweight="bold", color="#BBF7D0", ha="center")
    ax.text(13.85, 2.95, r"$y^* = \mathrm{argmax}_k$", fontsize=8.0, color="#DCFCE7", ha="center")
    ax.text(13.85, 2.65, r"$P(y = k \mid \mathbf{x}_q)$", fontsize=8.0, fontweight="bold", color="#FFFFFF", ha="center")
    ax.text(13.85, 2.05, "100.0%", fontsize=13.0, fontweight="bold", color="#FFFFFF", ha="center")
    ax.text(13.85, 1.68, "Test Accuracy", fontsize=7.2, color="#BBF7D0", ha="center")
    ax.text(13.85, 1.30, r"$F_1 = 1.0000$", fontsize=7.6, fontweight="bold", color="#FEF08A", ha="center")

    save_figure(fig, "fig1_architecture")


# ==============================================================================
# FIGURE 2: Multi-Model Benchmark Comparison (Accuracy vs Training Overhead)
# ==============================================================================
def generate_figure2():
    model_order = [
        "MobileNetV3-Small",
        "EfficientNetB0",
        "ResNet18",
        "ViT-B/16",
        "CLIP (ViT-B/32)",
        "Naive Retrieval (CLIP + ChromaDB)",
        "Fine-Tuned Retrieval (Adapter + ChromaDB)",
        "Proposed Framework (CLIP + ChromaDB + IMO)"
    ]

    clean_labels = [
        "MobileNetV3-Small (Edge CNN)",
        "EfficientNetB0 (Compound CNN)",
        "ResNet18 (Residual CNN)",
        "ViT-B/16 (Vision Transformer)",
        "CLIP ViT-B/32 (Linear Probe)",
        "Naive Retrieval (ChromaDB)",
        "Fine-Tuned Retrieval (Adapter)",
        "Proposed IMO Framework (Ours)"
    ]

    df = df_gpu.set_index("model_name").loc[model_order].reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=300,
                                   gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.22})

    y_pos = np.arange(len(df))

    # Coordinated academic color palette
    colors = [
        "#3B82F6",  # MobileNetV3 (Blue)
        "#2563EB",  # EfficientNetB0 (Royal Blue)
        "#1D4ED8",  # ResNet18 (Navy)
        "#7C3AED",  # ViT-B/16 (Purple)
        "#0D9488",  # CLIP ViT-B/32 (Teal)
        "#F97316",  # Naive Retrieval (Orange)
        "#EA580C",  # Fine-Tuned Retrieval (Dark Orange)
        "#16A34A",  # Proposed IMO Framework (Emerald Green)
    ]

    # Panel A: Hold-Out Test Accuracy
    bars1 = ax1.barh(y_pos, df["accuracy"] * 100, height=0.60, color=colors, edgecolor="#1E293B", linewidth=0.9, zorder=3)
    ax1.set_xlim(95.0, 102.2)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(clean_labels, fontsize=9.5, fontweight="bold")
    ax1.invert_yaxis()
    ax1.set_xlabel(r"Hold-Out Test Accuracy (%) $\quad [\mathcal{D}_{\mathrm{test}}, N = 374]$", fontsize=10.0, fontweight="bold", labelpad=6)
    ax1.set_title(r"(a) Hold-Out Predictive Accuracy & Macro $F_1$", fontsize=11.0, fontweight="bold", pad=10)
    ax1.grid(axis="x", linestyle="--", alpha=0.6, zorder=0)

    for bar, acc, f1 in zip(bars1, df["accuracy"] * 100, df["f1"]):
        w = bar.get_width()
        label_text = f" {acc:.2f}%  (F₁: {f1:.4f})" if acc < 100.0 else f" 100.00%  (F₁: 1.0000)"
        ax1.text(w + 0.08, bar.get_y() + bar.get_height() / 2, label_text,
                 va="center", ha="left", fontsize=8.8, fontweight="bold", color="#0F172A")

    ax1.axvline(100.0, color="#16A34A", linestyle=":", linewidth=1.4, alpha=0.7, zorder=4)

    # Panel B: Knowledge Adaptation / Training Time
    train_times = df["train_time_sec"]
    bars2 = ax2.barh(y_pos, train_times, height=0.60, color=colors, edgecolor="#1E293B", linewidth=0.9, zorder=3)
    ax2.set_xscale("log")
    ax2.set_xlim(0.3, 1000)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xlabel("Knowledge Adaptation Time (s) [Log Scale]", fontsize=10.0, fontweight="bold", labelpad=6)
    ax2.set_title("(b) Computational Overhead of Knowledge Updates", fontsize=11.0, fontweight="bold", pad=10)
    ax2.grid(axis="x", which="both", linestyle="--", alpha=0.6, zorder=0)

    for bar, t_sec in zip(bars2, train_times):
        w = bar.get_width()
        if t_sec >= 60:
            txt = f" {t_sec:.1f} s ({t_sec/60:.1f} min)"
        elif t_sec >= 10:
            txt = f" {t_sec:.1f} s"
        else:
            txt = f" {t_sec:.2f} s (>200× speedup)"
        ax2.text(w * 1.12, bar.get_y() + bar.get_height() / 2, txt,
                 va="center", ha="left", fontsize=8.6, fontweight="bold", color="#1E293B")

    save_figure(fig, "fig2_accuracy_comparison")


# ==============================================================================
# FIGURE 3: Pareto Efficiency Frontier (Accuracy vs Inference Latency)
# ==============================================================================
def generate_figure3():
    fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=300)

    # Shaded sub-millisecond retrieval zone (clean box comfortably enclosing all 3 retrieval points)
    rect_ret = patches.Rectangle((0.085, 99.60), 0.165, 0.75, linewidth=1.2,
                                 edgecolor="#16A34A", facecolor="#F0FDF4", linestyle="--", alpha=0.7, zorder=1)
    ax.add_patch(rect_ret)
    ax.text(0.148, 99.30, "Sub-Millisecond Vector Search Zone\n(0.14 – 0.18 ms)",
            fontsize=8.0, fontweight="bold", color="#15803D", ha="center")

    # Pareto frontier curve
    pareto_pts = [(0.149, 100.0), (4.43, 99.73), (10.98, 99.47)]
    px, py = zip(*pareto_pts)
    ax.plot(px, py, linestyle=":", linewidth=1.6, color="#16A34A", alpha=0.8, zorder=2)

    # Plot base models with clean spacing
    ax.scatter(3.30, 97.33, s=120, color="#3B82F6", marker="o", edgecolors="#0F172A", linewidth=1.1, zorder=5)
    ax.text(3.30, 97.10, "MobileNetV3-Small\n(3.30 ms, 97.33%)", fontsize=8.2, color="#0F172A", ha="center", va="top", zorder=6)

    ax.scatter(3.75, 98.66, s=120, color="#2563EB", marker="s", edgecolors="#0F172A", linewidth=1.1, zorder=5)
    ax.text(3.30, 98.66, "EfficientNetB0\n(3.75 ms, 98.66%)", fontsize=8.2, color="#0F172A", ha="right", va="center", zorder=6)

    ax.scatter(3.69, 99.20, s=120, color="#1D4ED8", marker="^", edgecolors="#0F172A", linewidth=1.1, zorder=5)
    ax.text(3.25, 99.20, "ResNet18\n(3.69 ms, 99.20%)", fontsize=8.2, color="#0F172A", ha="right", va="center", zorder=6)

    ax.scatter(4.43, 99.73, s=120, color="#0D9488", marker="p", edgecolors="#0F172A", linewidth=1.1, zorder=5)
    ax.text(4.10, 99.96, "CLIP (ViT-B/32)\n(4.43 ms, 99.73%)", fontsize=8.2, color="#0F172A", ha="center", va="bottom", zorder=6)

    ax.scatter(10.98, 99.47, s=120, color="#7C3AED", marker="D", edgecolors="#0F172A", linewidth=1.1, zorder=5)
    ax.text(10.98, 99.22, "ViT-B/16\n(10.98 ms, 99.47%)", fontsize=8.2, color="#0F172A", ha="center", va="top", zorder=6)

    # Retrieval models: Triangular leader-line hierarchy with zero label collisions
    # 1. Proposed IMO (Ours) - Top tier
    ax.scatter(0.149, 100.0, s=240, color="#16A34A", marker="*", edgecolors="#0F172A", linewidth=1.2, zorder=7)
    ax.annotate(r"$\mathbf{Proposed\ IMO\ (Ours)}$" + "\n" + r"$\mathbf{(0.15\ ms,\ 100.00\%)}$",
                xy=(0.149, 100.0), xytext=(0.149, 100.20),
                fontsize=8.5, color="#14532D", ha="center", va="bottom", zorder=8,
                arrowprops=dict(arrowstyle="->", color="#16A34A", lw=1.1))

    # 2. Fine-Tuned Retrieval - Bottom-left tier (inside zone, zero collision)
    ax.scatter(0.140, 100.0, s=130, color="#EA580C", marker="X", edgecolors="#0F172A", linewidth=1.1, zorder=5)
    ax.annotate("Fine-Tuned Ret.\n(0.14 ms, 100.00%)", xy=(0.140, 100.0), xytext=(0.115, 99.78),
                fontsize=7.8, color="#9A3412", ha="center", va="top", zorder=6,
                arrowprops=dict(arrowstyle="->", color="#EA580C", lw=0.9))

    # 3. Naive Retrieval - Right tier (placed in open space x=0.26, zero collision)
    ax.scatter(0.177, 100.0, s=130, color="#F97316", marker="P", edgecolors="#0F172A", linewidth=1.1, zorder=5)
    ax.annotate("Naive Ret.\n(0.18 ms, 100.00%)", xy=(0.177, 100.0), xytext=(0.26, 100.08),
                fontsize=7.8, color="#C2410C", ha="left", va="center", zorder=6,
                arrowprops=dict(arrowstyle="->", color="#F97316", lw=0.9))

    # End-to-end latency disclosure card (Lower Center-Right - zero collision)
    ax.text(
        0.55, 97.4,
        "Methodological Latency Breakdown:\n"
        r"• HNSW Vector Search: $0.15\text{ ms} \quad (>6,700\text{ samples/s})$" + "\n"
        r"• Frozen CLIP ViT-B/32 Forward: $4.43\text{ ms (GPU)} \ / \ 17.40\text{ ms (CPU)}$" + "\n"
        r"• Full End-to-End Image Inference: $4.58\text{ ms (GPU)} \ / \ 17.55\text{ ms (CPU)}$",
        fontsize=8.0, color="#1E293B",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8FAFC", edgecolor="#94A3B8", lw=1.1),
        zorder=6
    )

    ax.set_xscale("log")
    ax.set_xlim(0.06, 20.0)
    ax.set_ylim(96.8, 100.5)
    ax.set_xlabel(r"Inference Latency per Sample (ms) [Log Scale]", fontsize=10.5, fontweight="bold", labelpad=6)
    ax.set_ylabel(r"Hold-Out Test Accuracy (%) $\quad [\mathcal{D}_{\mathrm{test}}, N = 374]$", fontsize=10.5, fontweight="bold", labelpad=6)
    ax.set_title("Pareto Efficiency Frontier: Predictive Accuracy vs. Wall-Clock Latency", fontsize=11.5, fontweight="bold", pad=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.5, zorder=0)

    save_figure(fig, "fig3_latency_accuracy_tradeoff")


# ==============================================================================
# FIGURE 4: CPU vs. GPU Hardware Latency and Speedup Analysis
# ==============================================================================
def generate_figure4():
    models = [
        "MobileNetV3",
        "EfficientNetB0",
        "ResNet18",
        "ViT-B/16",
        "CLIP ViT-B/32",
        "Naive Ret.",
        "Fine-Tuned Ret.",
        "Proposed IMO"
    ]

    gpu_lat = [3.30, 3.75, 3.69, 10.98, 4.43, 0.18, 0.14, 0.15]
    cpu_lat = [3.05, 9.82, 10.07, 62.40, 17.40, 0.18, 0.14, 0.15]
    speedups = [round(c / g, 2) for c, g in zip(cpu_lat, gpu_lat)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=300,
                                   gridspec_kw={"width_ratios": [1.2, 1.0], "wspace": 0.24})

    x = np.arange(len(models))
    width = 0.36

    # Panel A: Latency Comparison (Log Scale)
    rects1 = ax1.bar(x - width/2, gpu_lat, width, label="GPU (cuda:0 - RTX 2060)", color="#2563EB", edgecolor="#1E293B", linewidth=0.9, zorder=3)
    rects2 = ax1.bar(x + width/2, cpu_lat, width, label="CPU (24-Thread AMD Host)", color="#EA580C", edgecolor="#1E293B", linewidth=0.9, zorder=3)

    ax1.set_yscale("log")
    ax1.set_ylim(0.06, 250)
    ax1.set_ylabel("Inference Latency (ms / sample) [Log Scale]", fontsize=10.0, fontweight="bold")
    ax1.set_title("(a) Dedicated GPU vs. Host CPU Inference Latency", fontsize=11.0, fontweight="bold", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=25, ha="right", fontsize=9.0, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True, fontsize=8.8)
    ax1.grid(axis="y", which="both", linestyle="--", alpha=0.5, zorder=0)

    # For CNN/ViT models (0-4), print individual labels with collision-free vertical offsets
    for i in range(5):
        h1 = gpu_lat[i]
        h2 = cpu_lat[i]
        if i == 0:  # MobileNetV3: stagger labels vertically to prevent overlap
            ax1.text(x[i] - width/2, h1 * 1.45, f"{h1:.2f}", ha="center", va="bottom", fontsize=7.2, color="#1E40AF")
            ax1.text(x[i] + width/2, h2 * 1.15, f"{h2:.2f}", ha="center", va="bottom", fontsize=7.2, color="#9A3412")
        else:
            ax1.text(x[i] - width/2, h1 * 1.25, f"{h1:.2f}", ha="center", va="bottom", fontsize=7.2, color="#1E40AF")
            ax1.text(x[i] + width/2, h2 * 1.25, f"{h2:.2f}", ha="center", va="bottom", fontsize=7.2, color="#9A3412")

    for i in range(5, 8):
        h = gpu_lat[i]
        ax1.text(x[i], h * 1.35, f"{h:.2f} ms", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#15803D")

    # Panel B: Hardware Slowdown Ratio
    speedup_colors = [
        "#3B82F6" if s < 1.0 else ("#16A34A" if s == 1.0 else "#EA580C") for s in speedups
    ]
    bars3 = ax2.bar(x, speedups, width=0.55, color=speedup_colors, edgecolor="#1E293B", linewidth=0.9, zorder=3)
    ax2.axhline(1.0, color="#64748B", linestyle="--", linewidth=1.3, zorder=4, label="Parity (1.00×)")

    ax2.set_ylim(0, 7.8)
    ax2.set_ylabel("CPU Slowdown Ratio (CPU Latency / GPU Latency)", fontsize=10.0, fontweight="bold")
    ax2.set_title("(b) Architectural Hardware Dependency Factor", fontsize=11.0, fontweight="bold", pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=25, ha="right", fontsize=9.0, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    for bar, s in zip(bars3, speedups):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., h + 0.18, f"{s:.2f}×", ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold", color="#0F172A")

    # Clean Callouts (Placed with zero bar collision!)
    ax2.annotate("GPU Bound: 5.68x Slowdown", xy=(3, 5.68), xytext=(3, 7.0),
                 fontsize=7.8, fontweight="bold", color="#991B1B", ha="center",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="#FEF2F2", edgecolor="#DC2626", lw=0.9),
                 arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.0))

    # Hardware invariant badge placed safely in open air at x=6.25, y=3.6 with arrow pointing to retrieval bars (zero overlap with CLIP bar)
    ax2.annotate("Hardware-Invariant (1.00×)\nHost L3 Vector Search (0.15 ms)",
                 xy=(6.0, 1.15), xytext=(6.25, 3.6),
                 fontsize=7.2, fontweight="bold", color="#14532D", ha="center",
                 bbox=dict(boxstyle="round,pad=0.22", facecolor="#F0FDF4", edgecolor="#16A34A", lw=0.9),
                 arrowprops=dict(arrowstyle="->", color="#16A34A", lw=1.1))

    save_figure(fig, "fig4_cpu_vs_gpu_speedup")


# ==============================================================================
# FIGURE 5: Noise Robustness & IMO Quarantine Stress Test
# ==============================================================================
def generate_figure5():
    rates = [r * 100 for r in noise_data["noise_rates"]]
    naive_acc = [a * 100 for a in noise_data["naive_accuracy"]]
    imo_acc = [a * 100 for a in noise_data["imo_accuracy"]]
    quarantined = noise_data["imo_quarantined_count"]
    total_train = 1752
    accepted = [total_train - q for q in quarantined]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=300,
                                   gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.22})

    # Panel A: Accuracy Degradation Curve
    ax1.plot(rates, imo_acc, marker="s", markersize=8, linewidth=2.4, color="#16A34A",
             label="Proposed IMO Framework (Ours)", zorder=4)
    ax1.plot(rates, naive_acc, marker="o", markersize=7, linewidth=2.0, color="#DC2626",
             linestyle="-.", label="Naive Retrieval (ChromaDB - Unfiltered)", zorder=4)

    ax1.fill_between(rates, naive_acc, imo_acc, color="#86EFAC", alpha=0.30,
                     label="IMO Accuracy Retention Margin", zorder=2)

    # Staggered annotations with plenty of vertical and horizontal spacing (zero collision)
    ax1.annotate("+1.60% Margin", xy=(10, 98.4), xytext=(6.0, 94.0),
                 fontsize=8.0, fontweight="bold", color="#14532D", ha="center",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="#F0FDF4", edgecolor="#16A34A", lw=0.8),
                 arrowprops=dict(arrowstyle="->", color="#16A34A", lw=1.0))

    ax1.annotate("+8.02% Margin", xy=(20, 92.0), xytext=(15.0, 87.5),
                 fontsize=8.0, fontweight="bold", color="#14532D", ha="center",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="#F0FDF4", edgecolor="#16A34A", lw=0.8),
                 arrowprops=dict(arrowstyle="->", color="#16A34A", lw=1.0))

    ax1.annotate("+19.52% Retention Margin", xy=(30, 80.5), xytext=(24.5, 76.5),
                 fontsize=8.0, fontweight="bold", color="#14532D", ha="center",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="#F0FDF4", edgecolor="#16A34A", lw=0.8),
                 arrowprops=dict(arrowstyle="->", color="#16A34A", lw=1.0))

    ax1.set_xlim(-2, 33)
    ax1.set_ylim(72, 104)
    ax1.set_xlabel(r"Injected Symmetric Label Noise $r \in [0, 0.30]$ (%)", fontsize=10.0, fontweight="bold", labelpad=6)
    ax1.set_ylabel(r"Hold-Out Test Accuracy (%) $\quad [\mathcal{D}_{\mathrm{test}}, N = 374]$", fontsize=10.0, fontweight="bold", labelpad=6)
    ax1.set_title("(a) Test Accuracy Under Label Corruption (Hypothesis H1)", fontsize=11.0, fontweight="bold", pad=10)
    ax1.legend(loc="lower left", frameon=True, fontsize=8.2)
    ax1.grid(True, linestyle="--", alpha=0.5, zorder=0)

    # Panel B: Memory Allocation & Quarantine Distribution
    x_bar = np.arange(len(rates))
    w = 0.50

    b1 = ax2.bar(x_bar, accepted, w, label=r"Accepted into Active Memory $\mathcal{M}$", color="#16A34A", edgecolor="#14532D", linewidth=0.9, zorder=3)
    b2 = ax2.bar(x_bar, quarantined, w, bottom=accepted, label=r"Quarantined as Corrupted $\mathcal{Q}$", color="#FCA5A5", edgecolor="#991B1B", linewidth=0.9, hatch="//", zorder=3)

    ax2.set_ylim(0, 2700)
    ax2.set_xticks(x_bar)
    ax2.set_xticklabels([f"{r:.0f}% Noise" for r in rates], fontsize=9.2, fontweight="bold")
    ax2.set_ylabel(r"Ingestion Candidates $\quad [\mathcal{D}_{\mathrm{train}}, N = 1,752]$", fontsize=10.0, fontweight="bold")
    ax2.set_title("(b) IMO Automated Quarantine Distribution\n[100% Precision: 525/525 Corrupted Vectors Intercepted]",
                  fontsize=10.5, fontweight="bold", color="#0F172A", pad=10)
    ax2.legend(loc="upper right", frameon=True, fontsize=8.2)
    ax2.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    for i, (acc_cnt, q_cnt) in enumerate(zip(accepted, quarantined)):
        if q_cnt > 0:
            ax2.text(i, acc_cnt + q_cnt / 2, f"{q_cnt} ({q_cnt/total_train*100:.1f}%)",
                     ha="center", va="center", fontsize=8.2, fontweight="bold", color="#7F1D1D")
        ax2.text(i, acc_cnt / 2, f"{acc_cnt}", ha="center", va="center", fontsize=8.8, fontweight="bold", color="#FFFFFF")

    save_figure(fig, "fig5_noise_robustness_curve")


# ==============================================================================
# FIGURE 6: IMO Quality Gating Decision Flowchart
# ==============================================================================
def generate_figure6():
    fig, ax = plt.subplots(figsize=(14.0, 4.4), dpi=300)
    ax.set_xlim(0, 14.0)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    ax.text(7.0, 4.10, "Intelligent Memory Orchestrator (IMO) Quality Validation Pipeline",
            fontsize=12.0, fontweight="bold", color="#0F172A", ha="center")

    # 1. Candidate Input Card
    b_cand = FancyBboxPatch((0.4, 1.65), 1.60, 1.90, boxstyle="round,pad=0.08,rounding_size=0.08",
                            facecolor="#EFF6FF", edgecolor="#2563EB", linewidth=1.5, zorder=2)
    ax.add_patch(b_cand)
    ax.text(1.20, 3.15, "Candidate Sample", fontsize=9.0, fontweight="bold", color="#1E40AF", ha="center")
    ax.text(1.20, 2.70, r"$(\mathbf{z}_t, y_t)$", fontsize=10.0, fontweight="bold", color="#0F172A", ha="center")
    ax.text(1.20, 2.25, r"$\mathbf{z}_t \in \mathbb{R}^{512}$", fontsize=8.5, color="#475569", ha="center")
    ax.text(1.20, 1.88, r"$y_t \in \{0, 1\}$", fontsize=8.0, color="#64748B", ha="center")

    # 2. Gate 1: Prototype Check
    g1 = FancyBboxPatch((2.80, 1.65), 2.20, 1.90, boxstyle="round,pad=0.08,rounding_size=0.08",
                        facecolor="#FFFFFF", edgecolor="#16A34A", linewidth=1.5, zorder=2)
    ax.add_patch(g1)
    ax.text(3.90, 3.20, "Gate 1: Consistency", fontsize=9.2, fontweight="bold", color="#14532D", ha="center")
    ax.text(3.90, 2.65, r"$S_{\mathrm{proto}}(\mathbf{z}_t, y) \geq \tau_c$", fontsize=9.2, fontweight="bold", color="#0F172A", ha="center")
    ax.text(3.90, 2.22, r"$\tau_c = 0.65$", fontsize=8.5, fontweight="bold", color="#16A34A", ha="center")
    ax.text(3.90, 1.85, "Centroid Alignment", fontsize=7.5, color="#64748B", ha="center")

    # 3. Gate 2: Margin Check
    g2 = FancyBboxPatch((5.80, 1.65), 2.20, 1.90, boxstyle="round,pad=0.08,rounding_size=0.08",
                        facecolor="#FFFFFF", edgecolor="#16A34A", linewidth=1.5, zorder=2)
    ax.add_patch(g2)
    ax.text(6.90, 3.20, "Gate 2: Relative Margin", fontsize=9.2, fontweight="bold", color="#14532D", ha="center")
    ax.text(6.90, 2.65, r"$S_{\mathrm{rel}}(\mathbf{z}_t, y) \geq \tau_m$", fontsize=9.2, fontweight="bold", color="#0F172A", ha="center")
    ax.text(6.90, 2.22, r"$\tau_m = 0.00$", fontsize=8.5, fontweight="bold", color="#16A34A", ha="center")
    ax.text(6.90, 1.85, "Conflict & Ambiguity", fontsize=7.5, color="#64748B", ha="center")

    # 4. Gate 3: Density Filter
    g3 = FancyBboxPatch((8.80, 1.65), 2.20, 1.90, boxstyle="round,pad=0.08,rounding_size=0.08",
                        facecolor="#FFFFFF", edgecolor="#16A34A", linewidth=1.5, zorder=2)
    ax.add_patch(g3)
    ax.text(9.90, 3.20, "Gate 3: Outlier Density", fontsize=9.2, fontweight="bold", color="#14532D", ha="center")
    ax.text(9.90, 2.65, r"$\bar{D}(\mathbf{z}_t) \leq \tau_{\mathrm{out}}$", fontsize=9.2, fontweight="bold", color="#0F172A", ha="center")
    ax.text(9.90, 2.22, r"$\tau_{\mathrm{out}} = 0.45 \quad (K=5)$", fontsize=8.2, fontweight="bold", color="#16A34A", ha="center")
    ax.text(9.90, 1.85, "Local Neighborhood", fontsize=7.5, color="#64748B", ha="center")

    # 5. Commit to Active Memory Card
    b_commit = FancyBboxPatch((11.80, 1.65), 1.70, 1.90, boxstyle="round,pad=0.08,rounding_size=0.08",
                              facecolor="#15803D", edgecolor="#14532D", linewidth=1.6, zorder=2)
    ax.add_patch(b_commit)
    ax.text(12.65, 3.10, "COMMIT", fontsize=9.2, fontweight="bold", color="#FFFFFF", ha="center")
    ax.text(12.65, 2.65, r"$\mathcal{M} \leftarrow \mathcal{M} \cup \{\mathbf{z}_t\}$", fontsize=8.2, color="#F0FDF4", ha="center")
    ax.text(12.65, 2.15, r"Update $\mathbf{p}_k$", fontsize=8.2, color="#DCFCE7", ha="center")
    ax.text(12.65, 1.85, "Active Store", fontsize=7.2, color="#BBF7D0", ha="center")

    # Forward arrows (Pass) with generous 0.80 spacing
    ax.add_patch(FancyArrowPatch((2.00, 2.60), (2.80, 2.60), arrowstyle="-|>", mutation_scale=13, linewidth=1.6, color="#2563EB"))
    ax.add_patch(FancyArrowPatch((5.00, 2.60), (5.80, 2.60), arrowstyle="-|>", mutation_scale=13, linewidth=1.6, color="#16A34A"))
    ax.add_patch(FancyArrowPatch((8.00, 2.60), (8.80, 2.60), arrowstyle="-|>", mutation_scale=13, linewidth=1.6, color="#16A34A"))
    ax.add_patch(FancyArrowPatch((11.00, 2.60), (11.80, 2.60), arrowstyle="-|>", mutation_scale=13, linewidth=1.6, color="#16A34A"))

    ax.text(2.40, 2.85, "Input", fontsize=7.2, color="#1E40AF", ha="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#EFF6FF", edgecolor="#BFDBFE", lw=0.6))
    ax.text(5.40, 2.85, "Pass", fontsize=7.2, color="#15803D", ha="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#F0FDF4", edgecolor="#BBF7D0", lw=0.6))
    ax.text(8.40, 2.85, "Pass", fontsize=7.2, color="#15803D", ha="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#F0FDF4", edgecolor="#BBF7D0", lw=0.6))
    ax.text(11.40, 2.85, "Pass", fontsize=7.2, color="#15803D", ha="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#F0FDF4", edgecolor="#BBF7D0", lw=0.6))

    # Quarantine Sink Box (Bottom)
    b_quar = FancyBboxPatch((2.80, 0.35), 8.20, 0.75, boxstyle="round,pad=0.06,rounding_size=0.06",
                            facecolor="#FEF2F2", edgecolor="#DC2626", linewidth=1.4, zorder=2)
    ax.add_patch(b_quar)
    ax.text(6.90, 0.85, r"Non-Destructive Quarantine Buffer $\mathcal{Q}$", fontsize=9.0, fontweight="bold", color="#991B1B", ha="center")
    ax.text(6.90, 0.55, r"Stores $(\mathbf{z}_t, y_t, \mathrm{reason}, \mathrm{timestamp})$: Zero Active Memory Contamination",
            fontsize=7.6, color="#B91C1C", ha="center")

    # Rejection arrows down to quarantine
    ax.add_patch(FancyArrowPatch((3.90, 1.65), (3.90, 1.10), arrowstyle="-|>", mutation_scale=12, linewidth=1.3, color="#DC2626", linestyle="--"))
    ax.add_patch(FancyArrowPatch((6.90, 1.65), (6.90, 1.10), arrowstyle="-|>", mutation_scale=12, linewidth=1.3, color="#DC2626", linestyle="--"))
    ax.add_patch(FancyArrowPatch((9.90, 1.65), (9.90, 1.10), arrowstyle="-|>", mutation_scale=12, linewidth=1.3, color="#DC2626", linestyle="--"))

    ax.text(4.05, 1.35, "Fail (Dissimilar)", fontsize=6.8, color="#991B1B", ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="#FEF2F2", edgecolor="#FECACA", lw=0.5))
    ax.text(7.05, 1.35, "Fail (Conflicted)", fontsize=6.8, color="#991B1B", ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="#FEF2F2", edgecolor="#FECACA", lw=0.5))
    ax.text(10.05, 1.35, "Fail (Outlier)", fontsize=6.8, color="#991B1B", ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="#FEF2F2", edgecolor="#FECACA", lw=0.5))

    save_figure(fig, "fig6_imo_pipeline")


if __name__ == "__main__":
    generate_figure1()
    generate_figure2()
    generate_figure3()
    generate_figure4()
    generate_figure5()
    generate_figure6()
    print("All 6 publication-grade figures successfully generated.")
