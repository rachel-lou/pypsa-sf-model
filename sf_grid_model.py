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
    # 230kV Substations (High Voltage Transmission)
    'Martin': {
        'v_nom': 230,
        'x': -122.4159,  # Bayshore & Geneva (confirmed)
        'y': 37.7089,
        'note': 'Main 230kV entry point, 70% of SF power'
    },
    'Embarcadero': {
        'v_nom': 230,
        'x': -122.3892,  # Financial District (you identified as 230kV)
        'y': 37.7953,
        'note': '230kV substation serving Financial District'
    },
    'Potrero_Switchyard': {
        'v_nom': 230,
        'x': -122.3898,  # Near Potrero Power Station site
        'y': 37.7588,
        'note': '230kV switchyard, adjacent to Potrero Converter (Trans Bay Cable)'
    },
    
    # 115kV Substations (Sub-Transmission)
    'Mission': {
        'v_nom': 115,
        'x': -122.4102,  # 8th & Mission (failed Dec 2024)
        'y': 37.7775,
        'note': '115kV substation, failed December 21, 2024'
    },
    'Larkin': {
        'v_nom': 115,
        'x': -122.4194,  # Larkin St area 
        'y': 37.7825,
        'note': '115kV substation, downtown'
    },
    'Bayshore': {
        'v_nom': 115,
        'x': -122.4035,  # Bayshore Blvd 
        'y': 37.7420,
        'note': '115kV substation, southern SF'
    },
    'HuntersPoint': {
        'v_nom': 115,
        'x': -122.3778, 
        'y': 37.7306,
        'note': '115kV substation, southeast SF'
    },
}

for name, props in buses.items():
    network.add("Bus", name, v_nom=props['v_nom'], x=props['x'], y=props['y'])

# =====================================================================
# GENERATORS - Power sources serving SF
# =====================================================================

# Hetch Hetchy hydropower (SFPUC) - enters at Martin
network.add("Generator", "ExternalGridMartin",
            bus="Martin",
            p_nom=1500,
            marginal_cost=50,
            carrier="external")
print("  • External Grid @ Martin: 1,500 MW (230kV entry point)")

# Trans Bay Cable at Potrero Converter
# Note: The 200kV DC converter connects to 230kV AC switchyard
network.add("Generator", "TransBayCable",
            bus="Potrero_Switchyard",
            p_nom=400,  # 400 MW underwater cable from Pittsburg
            marginal_cost=40,
            carrier="cable")
print("  • Trans Bay Cable @ Potrero: 400 MW (via DC converter)")

# Backup generation at critical substations
network.add("Generator", "MissionBackup",
            bus="Mission",
            p_nom=50,
            marginal_cost=200,
            carrier="diesel")
print("  • Emergency Backup @ Mission: 50 MW (diesel)")

network.add("Generator", "EmbarcaderoBackup",
            bus="Embarcadero",
            p_nom=30,
            marginal_cost=200,
            carrier="diesel")
print("  • Emergency Backup @ Embarcadero: 30 MW (diesel)")

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
    'Mission': 200,        # SoMa/Mission -
    'Larkin': 150,        # DOwntown/ Civic center
    'Embarcadero': 180,    # Financial District - 24K residential, 3K business
    'Potrero_Switchyard': 100,  # Potrero/Dogpatch
    'Bayshore': 120,      # Southern neighborhoods
    'HuntersPoint': 80,   # Southeast SF
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
     # 230kV 
    ('Martin', 'Potrero_Switchyard', 12, 600, 0.008, 0.08, '230kV-UG'),  # Underground
    ('Martin', 'Embarcadero', 10, 500, 0.008, 0.08, '230kV-UG'),         # Underground
    
    # 230kV to 115kV connections (transformers modeled as lines)
    ('Martin', 'Mission', 15, 400, 0.01, 0.10, '230/115kV'),  # Step-down
    ('Martin', 'Bayshore', 8, 350, 0.008, 0.08, '230/115kV'), # Step-down
    ('Embarcadero', 'Larkin', 3, 300, 0.005, 0.04, '230/115kV'),  # Step-down
    ('Potrero_Switchyard', 'HuntersPoint', 5, 300, 0.006, 0.05, '230/115kV'),  # Step-down
    
    # 115kV Sub-transmission Network
    ('Mission', 'Larkin', 3, 250, 0.005, 0.03, '115kV-UG'),    # Underground, downtown
    ('Mission', 'Bayshore', 6, 200, 0.007, 0.06, '115kV-UG'),  # Underground
    ('Larkin', 'Bayshore', 8, 180, 0.008, 0.07, '115kV-UG'),   # Underground
    ('Bayshore', 'HuntersPoint', 7, 200, 0.008, 0.06, '115kV-UG'),  # Underground
    
    # Cross-connections 
    ('Mission', 'Potrero_Switchyard', 5, 200, 0.006, 0.05, '115kV-UG'),  # Emergency path
]

for i, (bus0, bus1, length, s_nom, r, x, line_type) in enumerate(lines):
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
ax4 = fig.add_subplot(gs[2, 1])

max_loadings = line_loading_pct.max(axis=0)
peak_hours = line_loading_pct.idxmax(axis=0).apply(lambda x: x.hour)

colors = ['red' if x > 80 else 'orange' if x > 60 else 'green' 
          for x in max_loadings]

bars = ax4.barh(range(len(max_loadings)), max_loadings, color=colors)

# Add peak hour annotations
for i, (loading, hour) in enumerate(zip(max_loadings, peak_hours)):
    ax4.text(loading + 2, i, f"{hour}h", va='center', fontsize=7)

ax4.set_yticks(range(len(network.lines)))
ax4.set_yticklabels(line_names_short, fontsize=7)
ax4.set_xlabel("Maximum Loading (%)", fontsize=10)
ax4.set_title("Peak Line Loading\n(with hour of occurrence)", fontsize=11, fontweight='bold')
ax4.axvline(x=100, color='red', linestyle='--', linewidth=2, label='Capacity')
ax4.axvline(x=80, color='orange', linestyle='--', linewidth=2, label='High threshold')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3, axis='x')

# =====================================================================
# PLOT 5: Load Profile vs Generation - FIXED with reserve
# =====================================================================

ax5 = fig.add_subplot(gs[2, 2])

# Calculate totals
total_gen = network.generators_t.p.sum(axis=1)
total_load = network.loads_t.p.sum(axis=1)

# Plot
hours_array = network.snapshots.hour
ax5.plot(hours_array, total_gen, linewidth=3, label='Total Generation', 
         color='blue', marker='o', markersize=4)
ax5.plot(hours_array, total_load, linewidth=3, label='Total Load', 
         color='red', linestyle='--', marker='s', markersize=4)

# Fill reserve area
ax5.fill_between(hours_array, total_load, total_gen, 
                 where=(total_gen >= total_load),
                 alpha=0.3, color='green', label='Reserve Margin')

ax5.set_title("Generation vs Load\n(with Reserve)", fontsize=11, fontweight='bold')
ax5.set_xlabel("Hour of Day", fontsize=10)
ax5.set_ylabel("Power (MW)", fontsize=10)
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)
ax5.set_xlim(0, 23)

# Add annotations for min and max
min_load_idx = total_load.idxmin()
max_load_idx = total_load.idxmax()
ax5.annotate(f'Min Load\n{total_load.min():.0f} MW', 
            xy=(min_load_idx.hour, total_load.min()),
            xytext=(min_load_idx.hour + 2, total_load.min() - 30),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=8, color='red')
ax5.annotate(f'Peak Load\n{total_load.max():.0f} MW', 
            xy=(max_load_idx.hour, total_load.max()),
            xytext=(max_load_idx.hour - 4, total_load.max() + 30),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=8, color='red')

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