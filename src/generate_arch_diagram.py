"""
generate_arch_diagram.py
========================
Generates a high-resolution vector PDF and PNG block diagram
illustrating the 5G O-RAN Near-RT RIC 3-Pillar xApp Architecture.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_architecture_diagram():
    fig, ax = plt.subplots(figsize=(8.5, 2.9), dpi=300)
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 4.0)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Professional IEEE Color Palette
    c_ran = '#e2e8f0'      # Slate (gNB Block)
    c_p1  = '#fef3c7'      # Amber (Pillar 1)
    c_p3  = '#dcfce7'      # Emerald (Pillar 3)
    c_p2  = '#f3e8ff'      # Purple (Pillar 2)
    c_ric = '#fafafa'      # RIC Box Background
    c_border = '#334155'   # Border line color
    
    # 1. Outer Container: O-RAN Near-RT RIC
    ric_box = patches.Rectangle((2.1, 0.2), 7.8, 3.5, ec="#475569", fc=c_ric, linestyle='--', lw=1.3)
    ax.add_patch(ric_box)
    ax.text(6.0, 3.45, "O-RAN Near-Real-Time RIC (xApp Container)", ha='center', va='center', fontsize=9.5, fontweight='bold', color="#0f172a")
    
    # 2. Left Block: 5G NR gNB
    gnb_box = patches.Rectangle((0.1, 0.7), 1.6, 2.4, ec=c_border, fc=c_ran, lw=1.2)
    ax.add_patch(gnb_box)
    ax.text(0.9, 2.50, "5G NR gNB\n& UE Mobility", ha='center', va='center', fontsize=8.5, fontweight='bold')
    ax.text(0.9, 1.35, "RSRP Telemetry\n" + r"$\mathbf{R} \in \mathbb{R}^{50}$", ha='center', va='center', fontsize=7.5, color='#334155')
    
    # Arrow E2 Interface
    ax.annotate('', xy=(2.2, 1.9), xytext=(1.8, 1.9),
                arrowprops=dict(facecolor='#0284c7', edgecolor='#0284c7', width=1.5, headwidth=5))
    ax.text(2.0, 2.18, "E2SM-KPM", ha='center', va='center', fontsize=7, fontweight='bold', color='#0369a1')
    
    # 3. Pillar 1: Kinematic Physics Layer
    p1_box = patches.Rectangle((2.3, 0.7), 2.2, 2.4, ec=c_border, fc=c_p1, lw=1.2)
    ax.add_patch(p1_box)
    ax.text(3.4, 2.50, "Pillar 1: Kinematic\nPhysics Layer", ha='center', va='center', fontsize=8.5, fontweight='bold')
    ax.text(3.4, 1.35, r"$v_t = \frac{dr}{dt}$" + "\n" + r"$a_t = \frac{d^2r}{dt^2}$" + "\nEnergies (" + r"$E_v, E_a$" + ")\n" + r"$\rightarrow \mathbf{X}_{\mathrm{kin}} \in \mathbb{R}^{64}$", 
            ha='center', va='center', fontsize=7.5, color='#78350f')
    
    # Arrow P1 -> P3
    ax.annotate('', xy=(4.8, 1.9), xytext=(4.6, 1.9),
                arrowprops=dict(facecolor='#334155', edgecolor='#334155', width=1.2, headwidth=4))
    
    # 4. Pillar 3: TinyML Quantized Engine
    p3_box = patches.Rectangle((4.9, 0.7), 2.3, 2.4, ec=c_border, fc=c_p3, lw=1.2)
    ax.add_patch(p3_box)
    ax.text(6.05, 2.50, "Pillar 3: TinyML\nQuantized Engine", ha='center', va='center', fontsize=8.5, fontweight='bold')
    ax.text(6.05, 1.30, "INT8 MLP (56.8% RAM " + r"$\downarrow$" + ")\nPruned DT (0.096 ms)\n" + r"$\rightarrow \text{Prob. } \hat{p} \in [0,1]$", 
            ha='center', va='center', fontsize=7.0, color='#14532d')
    
    # Arrow P3 -> P2
    ax.annotate('', xy=(7.5, 1.9), xytext=(7.3, 1.9),
                arrowprops=dict(facecolor='#334155', edgecolor='#334155', width=1.2, headwidth=4))
    
    # 5. Pillar 2: DAT Cost Engine
    p2_box = patches.Rectangle((7.6, 0.7), 2.1, 2.4, ec=c_border, fc=c_p2, lw=1.2)
    ax.add_patch(p2_box)
    ax.text(8.65, 2.50, "Pillar 2: DAT\nCost Engine", ha='center', va='center', fontsize=8.5, fontweight='bold')
    ax.text(8.65, 1.30, r"$\mathcal{C}(P_{\mathrm{th}}) = \mathrm{FP} + 5\mathrm{FN}$" + "\nOpt. " + r"$P_{\mathrm{th}}^* = 0.35$" + "\n81.9% RLF Drop " + r"$\downarrow$" + "\n" + r"$\rightarrow \text{HO Decision}$", 
            ha='center', va='center', fontsize=7.0, color='#581c87')
    
    # Arrow P2 -> Out
    ax.annotate('', xy=(10.1, 1.9), xytext=(9.8, 1.9),
                arrowprops=dict(facecolor='#15803d', edgecolor='#15803d', width=1.5, headwidth=5))
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "figures")
    os.makedirs(output_dir, exist_ok=True)
    
    pdf_path = os.path.join(output_dir, "fig0_system_architecture.pdf")
    png_path = os.path.join(output_dir, "fig0_system_architecture.png")
    
    plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.02)
    plt.savefig(png_path, bbox_inches='tight', pad_inches=0.02, dpi=300)
    plt.close()
    
    print(f"[OK] Generated high-resolution architecture diagram:")
    print(f"  - PDF: {pdf_path}")
    print(f"  - PNG: {png_path}")

if __name__ == "__main__":
    create_architecture_diagram()
