"""
San Francisco Grid Security-Constrained Linear Optimal Power Flow (SCLOPF) Model

This PyPSA model simulates the San Francisco electrical grid with a focus on 
security-constrained optimal power flow to ensure system reliability during 
branch outages, inspired by the December 2024 Mission substation fire incident.

Based on the PyPSA SCLOPF example: https://pypsa.readthedocs.io/en/latest/examples/scigrid-sclopf.html
"""

import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create the network
network = pypsa.Network()
network.name = "San Francisco Grid"

# Define snapshots (hourly for one day)
network.set_snapshots(pd.date_range("2024-12-21 00:00", "2024-12-21 23:00", freq="h"))

# Define carriers to avoid warnings
network.add("Carrier", "AC")
network.add("Carrier", "hydro")
network.add("Carrier", "external")
network.add("Carrier", "cable")
network.add("Carrier", "diesel")
# =====================================================================
# BUSES - ACTUAL SF Substations from public records
# =====================================================================
# Sources: CPUC filings, PG&E website, CEC database
buses = {
    # Major 230kV/115kV Substation (70% of SF power goes through here)
    'Martin': {
        'v_nom': 230,  # Both 230kV and 115kV
        'x': -122.4159,  # Bayshore & Geneva (Brisbane/Daly City border)
        'y': 37.7089,
        'note': '70% of SF power, main entry point'
    },
    
    # Failed during December 2024 outage
    'Mission': {
        'v_nom': 115,
        'x': -122.4102,  # 8th & Mission Streets
        'y': 37.7775,
        'note': 'Failed Dec 21, 2024 - 130K customers affected'
    },
    
    # Major hub in Potrero Hill
    'Potrero': {
        'v_nom': 115,
        'x': -122.3953,  # 23rd Street, Potrero Hill
        'y': 37.7588,
        'note': 'Major distribution hub, near former power plant'
    },
    
    # Financial District
    'Embarcadero': {
        'v_nom': 115,
        'x': -122.3892,  # Near Embarcadero/Financial District
        'y': 37.7953,
        'note': 'Serves 24K residential, 3K business accounts'
    },
    
    # Southeast SF
    'HuntersPoint': {
        'v_nom': 115,
        'x': -122.3778,  # Evans Ave, Southeast SF
        'y': 37.7306,
        'note': 'Trans Bay Cable landing point, rebuilt 2006'
    },
    
    # New switching station (under construction/planned)
    'Egbert': {
        'v_nom': 230,
        'x': -122.3908,  # 1755 Egbert Avenue
        'y': 37.7289,
        'note': 'New backup for Martin Substation (planned/under construction)'
    },
    
    # Northern transmission entry (inferred from grid topology)
    'Jefferson': {
        'v_nom': 230,
        'x': -122.4700,  # Western SF, connects to Martin
        'y': 37.7800,
        'note': 'Northern 230kV entry point'
    },
}

for name, props in buses.items():
    network.add("Bus", name, v_nom=props['v_nom'], x=props['x'], y=props['y'])

# =====================================================================
# GENERATORS - Power sources serving SF
# =====================================================================

# Hetch Hetchy hydropower (SFPUC) - enters at Martin
network.add("Generator", "HetchHetchy",
            bus="Martin",
            p_nom=400,  # MW capacity
            marginal_cost=10,
            carrier="hydro")

# External PG&E grid connection
network.add("Generator", "ExternalGrid_Martin",
            bus="Martin",
            p_nom=1200,
            marginal_cost=50,
            carrier="external")

# Northern entry point
network.add("Generator", "ExternalGrid_Jefferson",
            bus="Jefferson",
            p_nom=300,
            marginal_cost=55,
            carrier="external")

# Trans Bay Cable (400 MW underwater cable from Pittsburg) https://en.wikipedia.org/wiki/Trans_Bay_Cable
# Lands at Hunters Point
network.add("Generator", "TransBayCable",
            bus="HuntersPoint",
            p_nom=400,
            marginal_cost=40,
            carrier="cable")

# Emergency backup at Mission
network.add("Generator", "MissionBackup",
            bus="Mission",
            p_nom=50,
            marginal_cost=200,
            carrier="diesel")


# =====================================================================
# LOADS - Electricity demand. Based on known service areas
# =====================================================================

# Create time-varying load profiles (simplified daily pattern)
hours = network.snapshots
# Peak demand in evening, lower at night. 
# (hours.hour - 6) shifts the sin wave peak to correspond with
# residential peak when people return home, cook dinner, use appliances
# Minimum at 6 AM: Lowest demand during early morning hours
# load_profile = .6 + 0.4 * np.sin((hours.hour - 6) * np.pi / 12)
# Range: 50% to 100% of base load: Typical daily variation for urban areas
# load_profile = np.maximum(load_profile, 0.5)
def realistic_load_profile(hours):
    """Create realistic daily load profile"""
    profile = np.zeros(len(hours))
    
    for i, hour in enumerate(hours.hour):
        if 0 <= hour < 6:  # Overnight - low demand
            profile[i] = 0.55 + 0.05 * np.sin(hour * np.pi / 6)
        elif 6 <= hour < 9:  # Morning ramp-up
            profile[i] = 0.60 + 0.15 * (hour - 6) / 3
        elif 9 <= hour < 16:  # Midday plateau
            profile[i] = 0.75 + 0.08 * np.sin((hour - 12.5) * np.pi / 7)
        elif 16 <= hour < 20:  # Evening peak
            profile[i] = 0.83 + 0.17 * np.sin((hour - 16) * np.pi / 4)
        else:  # Evening decline (20-24)
            profile[i] = 0.75 - 0.20 * (hour - 20) / 4
    
    return profile

load_profile = realistic_load_profile(hours)

loads = {
    'Mission': 200,        # SoMa/Mission - 130K customers affected in outage
    'Potrero': 120,        # Potrero/Dogpatch distribution
    'Embarcadero': 150,    # Financial District - 24K residential, 3K business
    'HuntersPoint': 80,    # Southeast SF
    'Egbert': 100,         # Southern neighborhoods served via Egbert
}

for name, base_load in loads.items():
    p_set = base_load * load_profile
    network.add("Load", f"Load_{name}",
                bus=name,
                p_set=p_set)

# =====================================================================
# TRANSMISSION LINES - Connections between substations
# =====================================================================
# (bus0, bus1, length (km), s_nom (apparent power rating in megavolt-ampere MVA), r(resistance per km in ohm/km), x (reactance per km  in ohm/km))

lines = [
    # 230kV transmission backbone
    ('Jefferson', 'Martin', 8, 500, 0.008, 0.08),  # Northern entry to Martin
    ('Martin', 'Egbert', 10, 400, 0.01, 0.10),     # Martin to Egbert (new project)
    
    # Martin to 115kV substations (step-down transformers modeled as lines)
    ('Martin', 'Mission', 15, 350, 0.01, 0.10),    # Critical path - failed Dec 2024
    ('Martin', 'Potrero', 18, 300, 0.012, 0.11),   # Martin to Potrero
    
    # Egbert connections (new reliability project)
    ('Egbert', 'Embarcadero', 8, 300, 0.008, 0.08),
    ('Egbert', 'HuntersPoint', 5, 250, 0.006, 0.06),
    
    # 115kV distribution network
    ('Mission', 'Embarcadero', 4, 250, 0.005, 0.04),    # Mission to FiDi
    ('Mission', 'Potrero', 5, 200, 0.006, 0.05),        # Mission to Potrero
    ('Potrero', 'HuntersPoint', 6, 200, 0.007, 0.06),   # Potrero to Hunters Point
    ('Embarcadero', 'Potrero', 5, 200, 0.006, 0.05),    # FiDi to Potrero
    
    # Trans Bay Cable connection
    ('HuntersPoint', 'Potrero', 6, 400, 0.006, 0.05),   # TBC to grid
]

for i, (bus0, bus1, length, s_nom, r, x) in enumerate(lines):
    network.add("Line", f"Line_{i}_{bus0}_{bus1}",
                bus0=bus0,
                bus1=bus1,
                length=length,  # km
                s_nom=s_nom,    # MVA rating
                r=r * length,   # Resistance (ohm)
                x=x * length,   # Reactance (ohm)
                capital_cost=0)

# =====================================================================
# ANALYSIS
# =====================================================================

print("=" * 70)
print("SAN FRANCISCO GRID SECURITY-CONSTRAINED OPTIMAL POWER FLOW")
print("=" * 70)
print(f"\nNetwork: {len(network.buses)} buses, {len(network.lines)} lines")
print(f"Generators: {len(network.generators)}")
print(f"Total load: {sum(loads.values()):.0f} MW (base)")
print("\n" + "=" * 70)

# First, run standard LOPF to establish baseline
print("\n1. Running standard Linear Optimal Power Flow (LOPF)...")
# Finds the optimal power dispatch for every hour (24 snapshots)
# Objective is to minimize the sum of generator_output * marginal_cost
# Constrained by:
#  power balance. generation = load at eevry bus every hour
#  generator limits. 0 <= p <=p_nom. power cannot exceed capacity
#  line limits. power flow <= rated capacity (s_nom)
#  power flow physics. kirchoff's laws
network.optimize(solver_name='highs')
print(f"   ✓ Objective value: ${network.objective:,.2f}")

print(f"\n   Generation dispatch:")
for gen in network.generators.index:
    gen_output = network.generators_t.p[gen].mean()
    print(f"     {gen}: {gen_output:.1f} MW (avg)")

# Line loadings
print(f"\n   Critical line loadings:")
line_loading = network.lines_t.p0.abs() / network.lines.s_nom * 100

# Focus on Martin lines (critical)
martin_lines = [line for line in network.lines.index if 'Martin' in line]
for line in martin_lines:
    max_load = line_loading[line].max()
    status = " HIGH" if max_load > 80 else " OK"
    print(f"     {line}: {max_load:.1f}% {status}")

# =====================================================================
# SECURITY-CONSTRAINED LOPF (SCLOPF)
# =====================================================================

print("\n" + "=" * 70)
print("2. Running Security-Constrained LOPF (SCLOPF)...")
print("=" * 70)

# Select Mission line to model outage
critical_lines = [line for line in network.lines.index if 'Mission' in line]

print(f"\n   Analyzing {len(critical_lines)} critical line outage scenarios:")
for line in critical_lines:
    print(f"     - {line}")

# Run SCLOPF
network.optimize.optimize_security_constrained(
    branch_outages=critical_lines,
    solver_name='highs'
)

print(f"\n   SCLOPF Objective value: ${network.objective:,.2f}")
print(f"\n   Security-constrained generation dispatch:")
for gen in network.generators.index:
    gen_output = network.generators_t.p[gen].mean()
    print(f"     {gen}: {gen_output:.1f} MW (avg)")

# Analyze the difference
print("\n" + "=" * 70)
print("3. COMPARISON: Standard LOPF vs Security-Constrained LOPF")
print("=" * 70)
print("  • SCLOPF ensures system survives any single line outage")
print("  • May require more expensive generation or different dispatch")

# =====================================================================
# VISUALIZATIONS
# =====================================================================

print("\n" + "=" * 70)
print("4. Generating visualizations...")
print("=" * 70)

fig, axes = plt.subplots( figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Network topology
ax1 = fig.add_subplot(gs[0, :])

# Extract coordinates
bus_coords = network.buses[['x', 'y']].values

# Plot lines first
for line_name in network.lines.index:
    bus0 = network.lines.loc[line_name, 'bus0']
    bus1 = network.lines.loc[line_name, 'bus1']
    x0, y0 = network.buses.loc[bus0, ['x', 'y']]
    x1, y1 = network.buses.loc[bus1, ['x', 'y']]
    
    ax1.plot([x0, x1], [y0, y1], 'b-', linewidth=2, alpha=0.6, zorder=1)

# Plot buses
for bus_name in network.buses.index:
    x, y = network.buses.loc[bus_name, ['x', 'y']]
    
    # Color code by voltage
    v_nom = network.buses.loc[bus_name, 'v_nom']
    color = 'red' if v_nom >= 230 else 'blue'
    size = 300 if v_nom >= 230 else 200
    
    ax1.scatter(x, y, s=size, c=color, alpha=0.7, edgecolors='black', 
                linewidths=2, zorder=2)
    
    # Add labels
    ax1.annotate(bus_name, (x, y), xytext=(5, 5), 
                textcoords='offset points', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

ax1.set_xlabel("Longitude (°W)", fontsize=11)
ax1.set_ylabel("Latitude (°N)", fontsize=11)
ax1.set_title("San Francisco Grid Topology\n(Red=230kV, Blue=115kV)", 
              fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='red', edgecolor='black', label='230 kV Substation'),
    Patch(facecolor='blue', edgecolor='black', label='115 kV Substation'),
]
ax1.legend(handles=legend_elements, loc='upper right')


# Plot 2: Generator dispatch over time
ax2 = fig.add_subplot(gs[1, :])

# Plot each generator
colors = ['green', 'orange', 'purple', 'cyan', 'red']
for i, gen in enumerate(network.generators.index):
    ax2.plot(network.snapshots.hour, network.generators_t.p[gen], 
             linewidth=2.5, label=gen, color=colors[i % len(colors)])

ax2.set_title("Generator Dispatch Profile (24 Hours - Dec 21, 2024)", 
              fontsize=13, fontweight='bold')
ax2.set_xlabel("Hour of Day", fontsize=11)
ax2.set_ylabel("Power Output (MW)", fontsize=11)
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 23)

# Plot 3: Line loading
ax3 = fig.add_subplot(gs[2, 0])

# Create heatmap of line loadings
line_loading_pct = network.lines_t.p0.abs() / network.lines.s_nom * 100
line_names_short = [name.replace('Line_', 'L').replace('_', '-') 
                   for name in network.lines.index]

im = ax3.imshow(line_loading_pct.T, aspect='auto', cmap='RdYlGn_r', 
                vmin=0, vmax=100, interpolation='nearest')
ax3.set_yticks(range(len(network.lines)))
ax3.set_yticklabels(line_names_short, fontsize=7)
ax3.set_xlabel("Hour of Day", fontsize=10)
ax3.set_title("Line Loading Heatmap (%)", fontsize=11, fontweight='bold')
ax3.set_xticks(range(0, 24, 3))
ax3.set_xticklabels(range(0, 24, 3))
# Add colorbar
cbar = plt.colorbar(im, ax=ax3)
cbar.set_label('Loading (%)', fontsize=9)

# Add contour at 80%
ax3.contour(line_loading_pct.T, levels=[80], colors='red', linewidths=2, alpha=0.5)


# Plot 4: Total load vs generation
# ax4 = fig.add_subplot(gs[2, 1])

# max_loadings = line_loading_pct.max(axis=0)
# print(line_loading_pct)
# peak_hours = line_loading_pct.snapshot

# colors = ['red' if x > 80 else 'orange' if x > 60 else 'green' 
#           for x in max_loadings]

# bars = ax4.barh(range(len(max_loadings)), max_loadings, color=colors)

# Add peak hour annotations
# for i, (loading, hour) in enumerate(zip(max_loadings, peak_hours)):
#     ax4.text(loading + 2, i, f"{hour}h", va='center', fontsize=7)

# ax4.set_yticks(range(len(network.lines)))
# ax4.set_yticklabels(line_names_short, fontsize=7)
# ax4.set_xlabel("Maximum Loading (%)", fontsize=10)
# ax4.set_title("Peak Line Loading\n(with hour of occurrence)", fontsize=11, fontweight='bold')
# ax4.axvline(x=100, color='red', linestyle='--', linewidth=2, label='Capacity')
# ax4.axvline(x=80, color='orange', linestyle='--', linewidth=2, label='High threshold')
# ax4.legend(fontsize=8)
# ax4.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('sf_grid_sclopf_analysis.png', dpi=300, bbox_inches='tight')
print("   Saved: sf_grid_sclopf_analysis.png")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print("\nThis model demonstrates how SCLOPF can help prevent outages like")
print("the December 2024 Mission substation failure by ensuring the system")
print("remains stable even when critical infrastructure components fail.")
print("=" * 70)