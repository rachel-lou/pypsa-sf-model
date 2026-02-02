"""
San Francisco Grid Model — Waymo Depot Expansion & New Site Placement

PHASE 1: Existing depot expansion
  • 3 known Waymo depots at current capacity
  • Test incremental expansions (add 20 vehicles at a time)
  • Find maximum feasible expansion before hitting grid limits

PHASE 2: New depot placement (greedy)
  • Starting from existing depots
  • Evaluate remaining candidate sites
  • Place depots greedily until hitting threshold

Known Waymo depots:
  • 201 Toland St (Bayshore): 117 vehicles, 36 chargers
  • 1155 Mission St (SoMa): estimated 60 vehicles
  • 14th Street (Mission): estimated 40 vehicles
"""

import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# PARAMETERS
# =====================================================================

VEHICLE_BATTERY_KWH  = 75
CHARGER_POWER_KW     = 30
CHARGING_EFF         = 0.90
DAILY_DRIVING_KM     = 250
ENERGY_PER_KM        = 0.20
CHARGERS_PER_VEHICLE = 0.31  # 36 chargers / 117 vehicles at Toland

OVERLOAD_THRESHOLD = 68.4 #percent

# =====================================================================
# PROFILES
# =====================================================================

def waymo_driving_pattern(hours):
    p = np.zeros(len(hours))
    for i, h in enumerate(hours.hour):
        if   h <  5: p[i] = 2.0
        elif h <  9: p[i] = 15.0 + 10.0 * np.sin((h-5)*np.pi/4)
        elif h < 11: p[i] = 12.0
        elif h < 14: p[i] = 14.0
        elif h < 17: p[i] = 13.0
        elif h < 21: p[i] = 18.0 + 7.0 * np.sin((h-17)*np.pi/4)
        else:        p[i] = 10.0 - 4.0*(h-21)/3
    return p

def sf_load_profile(hours):
    p = np.zeros(len(hours))
    for i, h in enumerate(hours.hour):
        if   h <  6: p[i] = 0.55 + 0.05*np.sin(h*np.pi/6)
        elif h <  9: p[i] = 0.60 + 0.15*(h-6)/3
        elif h < 16: p[i] = 0.75 + 0.08*np.sin((h-12.5)*np.pi/7)
        elif h < 20: p[i] = 0.83 + 0.17*np.sin((h-16)*np.pi/4)
        else:        p[i] = 0.75 - 0.20*(h-20)/4
    return p

# =====================================================================
# BASE NETWORK
# =====================================================================

def build_base_network():
    net = pypsa.Network()
    net.set_snapshots(pd.date_range("2024-12-21 00:00", "2024-12-21 23:00", freq="h"))
    for c in ["AC","external","cable","diesel","solar","Li-ion"]:
        net.add("Carrier", c)

    buses = {
        'Martin':             {'v_nom':230, 'x':-122.4159, 'y':37.7089},
        'Embarcadero':        {'v_nom':230, 'x':-122.3892, 'y':37.7953},
        'Potrero_Switchyard': {'v_nom':230, 'x':-122.3898, 'y':37.7588},
        'Mission':            {'v_nom':115, 'x':-122.4102, 'y':37.7775},
        'Larkin':             {'v_nom':115, 'x':-122.4194, 'y':37.7825},
        'Bayshore':           {'v_nom':115, 'x':-122.4035, 'y':37.7420},
        'HuntersPoint':       {'v_nom':115, 'x':-122.3778, 'y':37.7306},
    }
    for name, p in buses.items():
        net.add("Bus", name, v_nom=p['v_nom'], x=p['x'], y=p['y'], carrier="AC")

    net.add("Generator","ExternalGridMartin",   bus="Martin",            p_nom=1500, marginal_cost=50,  carrier="external")
    net.add("Generator","TransBayCable",        bus="Potrero_Switchyard",p_nom=200,  marginal_cost=40,  carrier="cable")
    net.add("Generator","MissionBackup",        bus="Mission",           p_nom=50,   marginal_cost=200, carrier="diesel")
    net.add("Generator","EmbarcaderoBackup",    bus="Embarcadero",       p_nom=30,   marginal_cost=200, carrier="diesel")

    solar = np.array([max(0, np.sin((h-6)*np.pi/12)) if 6<=h<=18 else 0 for h in range(24)])
    net.add("Generator","SunsetReservoir_Solar", bus="Mission", p_nom=5,     p_max_pu=solar, marginal_cost=0, carrier="solar")
    net.add("Generator","Distributed_PV",        bus="Larkin",  p_nom=38.77, p_max_pu=solar, marginal_cost=0, carrier="solar")

    hours   = net.snapshots
    lp      = sf_load_profile(hours)
    avg_mw  = 750
    load_dist = {
        'Mission':            0.22,
        'Larkin':             0.20,
        'Embarcadero':        0.18,
        'Potrero_Switchyard': 0.15,
        'Bayshore':           0.15,
        'HuntersPoint':       0.10,
    }
    for name, frac in load_dist.items():
        net.add("Load", f"Load_{name}", bus=name, p_set=avg_mw * frac * lp)

    lines = [
        ('Martin',            'Potrero_Switchyard', 12, 350, 0.008, 0.08),
        ('Martin',            'Embarcadero',        10, 300, 0.008, 0.08),
        ('Martin',            'Mission',            15, 300, 0.012, 0.10),
        ('Martin',            'Bayshore',            8, 280, 0.010, 0.08),
        ('Embarcadero',       'Larkin',              3, 250, 0.006, 0.04),
        ('Potrero_Switchyard','HuntersPoint',        5, 250, 0.007, 0.05),
        ('Mission',   'Larkin',          3, 180, 0.006, 0.03),
        ('Mission',   'Bayshore',        6, 170, 0.008, 0.06),
        ('Larkin',    'Bayshore',        8, 160, 0.009, 0.07),
        ('Bayshore',  'HuntersPoint',    7, 175, 0.008, 0.06),
        ('Mission',   'Potrero_Switchyard', 5, 130, 0.007, 0.05),
    ]
    for i, (b0, b1, length, s_nom, r, x) in enumerate(lines):
        net.add("Line", f"Line_{i}_{b0}_{b1}",
                bus0=b0, bus1=b1, length=length, s_nom=s_nom,
                r=r*length, x=x*length, capital_cost=0)

    return net

# =====================================================================
# DEPOT DEFINITIONS
# =====================================================================

# Known existing Waymo depots (LOCKED IN)
EXISTING_DEPOTS = {
    'Toland_St': {
        'bus': 'Bayshore',
        'current_vehicles': 117,
        'coords': (-122.3883, 37.7356),
        'type': 'existing'
    },
    'Mission_SoMa': {
        'bus': 'Mission',
        'current_vehicles': 60,
        'coords': (-122.4122, 37.7770),
        'type': 'existing'
    },
    '14th_Street': {
        'bus': 'Mission',
        'current_vehicles': 40,
        'coords': (-122.4194, 37.7677),
        'type': 'existing'
    },
}

# New candidate sites for greenfield depots
NEW_CANDIDATES = {
    'Hunters_Point':   {'bus':'HuntersPoint',       'coords':(-122.3778, 37.7306)},
    'Potrero_Hill':    {'bus':'Potrero_Switchyard', 'coords':(-122.3950, 37.7650)},
    'Embarcadero':     {'bus':'Embarcadero',        'coords':(-122.3892, 37.7953)},
    'Larkin_Civic':    {'bus':'Larkin',             'coords':(-122.4194, 37.7825)},
    'Martin_Bayshore': {'bus':'Martin',             'coords':(-122.4159, 37.7089)},
}

ALL_DEPOTS = {**EXISTING_DEPOTS, **NEW_CANDIDATES}

# =====================================================================
# DEPOT HELPERS
# =====================================================================

def add_depot(net, name, bus, n_vehicles):
    """Add depot with n_vehicles capacity."""
    hours    = net.snapshots
    drv_kwh  = waymo_driving_pattern(hours) * ENERGY_PER_KM
    
    n_chargers = int(n_vehicles * CHARGERS_PER_VEHICLE)
    charger_mw = (n_chargers * CHARGER_POWER_KW) / 1000
    battery_kwh = n_vehicles * VEHICLE_BATTERY_KWH

    batt_bus = f"{name}_batt"
    net.add("Bus", batt_bus, carrier="Li-ion")
    net.add("Store", f"{name}_store",
            bus=batt_bus, e_nom=battery_kwh,
            e_cyclic=True, e_initial=battery_kwh*0.5)
    net.add("Link", f"{name}_charger",
            bus0=bus, bus1=batt_bus,
            p_nom=charger_mw, efficiency=CHARGING_EFF)
    net.add("Load", f"{name}_driving",
            bus=batt_bus, p_set=(drv_kwh * n_vehicles)/1000)

def build_with_depot_config(depot_config):
    """
    depot_config = {'depot_name': n_vehicles, ...}
    """
    net = build_base_network()
    for name, n_veh in depot_config.items():
        bus = ALL_DEPOTS[name]['bus']
        add_depot(net, name, bus, n_veh)
    return net

def solve(depot_config):
    net = build_with_depot_config(depot_config)
    net.optimize(solver_name='highs')
    ll  = net.lines_t.p0.abs() / net.lines.s_nom * 100
    return net, net.objective, ll.max().max(), ll

# =====================================================================
# PHASE 1: EXPANSION ANALYSIS
# =====================================================================

print("=" * 80)
print("PHASE 1 — EXISTING DEPOT EXPANSION ANALYSIS")
print("=" * 80)

# Start with current capacities
base_config = {name: info['current_vehicles'] for name, info in EXISTING_DEPOTS.items()}
total_current = sum(base_config.values())

print(f"\nCurrent depot configuration:")
for name, n_veh in base_config.items():
    print(f"  • {name:20s} {n_veh:3d} vehicles at {EXISTING_DEPOTS[name]['bus']}")
print(f"  Total: {total_current} vehicles")

# Test baseline with current depots
net, obj, ml, ll = solve(base_config)
print(f"\nBaseline (current depots):")
print(f"  Cost: ${obj:,.0f} | Max loading: {ml:.1f} %")

# Test expansion for each depot independently
EXPANSION_INCREMENT = 20  # vehicles per step

expansion_results = {}

for depot_name in EXISTING_DEPOTS.keys():
    print(f"\n{'-' * 80}")
    print(f"Testing expansion: {depot_name}")
    print(f"{'-' * 80}")
    
    current_size = base_config[depot_name]
    max_feasible = current_size
    step = 0
    
    while True:
        trial_size = current_size + (step + 1) * EXPANSION_INCREMENT
        trial_config = base_config.copy()
        trial_config[depot_name] = trial_size
        
        _, _, ml, _ = solve(trial_config)
        
        status = "✓" if ml <= OVERLOAD_THRESHOLD else "✗"
        print(f"  Step {step+1}: +{(step+1)*EXPANSION_INCREMENT:3d} vehicles ({trial_size:3d} total) → {ml:6.2f} % {status}")
        
        if ml <= OVERLOAD_THRESHOLD:
            max_feasible = trial_size
            step += 1
        else:
            break
    
    expansion_capacity = max_feasible - current_size
    expansion_results[depot_name] = {
        'current': current_size,
        'max_feasible': max_feasible,
        'expansion': expansion_capacity,
    }
    print(f"  → Maximum: {max_feasible} vehicles (expansion: +{expansion_capacity})")

# Find optimal expansion strategy
print(f"\n{'=' * 80}")
print("EXPANSION SUMMARY")
print(f"{'=' * 80}")
for name, result in expansion_results.items():
    print(f"  {name:20s} current={result['current']:3d}  max={result['max_feasible']:3d}  expansion=+{result['expansion']:3d}")

# Use maximum feasible for all depots as starting point for Phase 2
expanded_config = {name: expansion_results[name]['max_feasible'] 
                   for name in EXISTING_DEPOTS.keys()}
total_expanded = sum(expanded_config.values())
total_expansion = total_expanded - total_current

print(f"\nOptimal expanded configuration:")
for name, n_veh in expanded_config.items():
    print(f"  • {name:20s} {n_veh:3d} vehicles (+{n_veh - base_config[name]:3d})")
print(f"  Total: {total_expanded} vehicles (+{total_expansion} expansion)")

# Verify expanded config
net_expanded, obj_expanded, ml_expanded, ll_expanded = solve(expanded_config)
print(f"\nExpanded depot performance:")
print(f"  Cost: ${obj_expanded:,.0f} | Max loading: {ml_expanded:.1f} %")

# =====================================================================
# PHASE 2: NEW DEPOT PLACEMENT (GREEDY)
# =====================================================================

print(f"\n{'=' * 80}")
print("PHASE 2 — NEW DEPOT PLACEMENT (GREEDY)")
print(f"{'=' * 80}")
print(f"Starting from expanded existing depots ({total_expanded} vehicles)")
print(f"Candidates: {len(NEW_CANDIDATES)}\n")

STANDARD_DEPOT_SIZE = 100  # New depots are 100-vehicle facilities

placed_new = []
remaining = set(NEW_CANDIDATES.keys())
history = [{
    'step': 0,
    'placed': None,
    'config': expanded_config.copy(),
    'objective': obj_expanded,
    'max_loading': ml_expanded,
    'line_loadings': ll_expanded,
    'network': net_expanded,
}]

step = 1
current_config = base_config.copy()

while remaining:
    print(f"STEP {step} — evaluating {len(remaining)} new candidates …")
    best_name, best_ml, best_result = None, float('inf'), None

    for cand in sorted(remaining):
        trial_config = current_config.copy()
        trial_config[cand] = STANDARD_DEPOT_SIZE
        
        net, obj, ml, ll = solve(trial_config)
        tag = " ← best" if ml < best_ml else ""
        print(f"    {cand:20s} bus={NEW_CANDIDATES[cand]['bus']:25s} max={ml:6.1f} %{tag}")
        
        if ml < best_ml:
            best_name, best_ml, best_result = cand, ml, (net, obj, ml, ll)

    if best_ml > OVERLOAD_THRESHOLD:
        print(f"\n  ✗ {best_name} would reach {best_ml:.1f} % > {OVERLOAD_THRESHOLD} % — stopping.\n")
        break

    net, obj, ml, ll = best_result
    current_config[best_name] = STANDARD_DEPOT_SIZE
    placed_new.append(best_name)
    remaining.remove(best_name)
    
    history.append({
        'step': step,
        'placed': best_name,
        'config': current_config.copy(),
        'objective': obj,
        'max_loading': ml,
        'line_loadings': ll,
        'network': net,
    })
    
    print(f"  ✓ Placed {best_name} ({STANDARD_DEPOT_SIZE} vehicles) → max loading {ml:.1f} %")
    print(f"    Total depots: {len(EXISTING_DEPOTS) + len(placed_new)} ({sum(current_config.values())} vehicles)\n")
    step += 1

print("=" * 80)
print(f"PHASE 2 COMPLETE — {len(placed_new)} new depots placed")
print("=" * 80)

# Calculate rejected candidates
remaining_rejected = remaining
trial_loadings = {}
if remaining_rejected:
    print("\nEvaluating rejected candidates …")
    for cand in sorted(remaining_rejected):
        trial_config = current_config.copy()
        trial_config[cand] = STANDARD_DEPOT_SIZE
        _, _, ml, _ = solve(trial_config)
        trial_loadings[cand] = ml
        print(f"    {cand:20s} → {ml:.1f} %  (rejected)")

# =====================================================================
# FINAL SUMMARY
# =====================================================================

print(f"\n{'=' * 80}")
print("FINAL CONFIGURATION SUMMARY")
print(f"{'=' * 80}")

print(f"\nEXISTING DEPOTS (expanded):")
for name in EXISTING_DEPOTS.keys():
    current = base_config[name]
    expanded = expanded_config[name]
    expansion = expanded - current
    print(f"  • {name:20s} {expanded:3d} vehicles (+{expansion:3d} expansion)")

print(f"\nNEW DEPOTS (greenfield):")
if placed_new:
    for name in placed_new:
        print(f"  • {name:20s} {STANDARD_DEPOT_SIZE:3d} vehicles (new)")
else:
    print(f"  (none — all rejected)")

total_final_vehicles = sum(current_config.values())
total_depots = len(EXISTING_DEPOTS) + len(placed_new)

print(f"\nTOTAL FLEET:")
print(f"  Depots:   {total_depots}")
print(f"  Vehicles: {total_final_vehicles}")
print(f"  Expansion from baseline: +{total_final_vehicles - total_current} vehicles")
print(f"  Final max line loading: {history[-1]['max_loading']:.1f} %")
print(f"  Final cost: ${history[-1]['objective']:,.0f}")

# =====================================================================
# VISUALIZATIONS
# =====================================================================

fig = plt.figure(figsize=(48, 18))
# make center column dominant so the network map can sit centered at the top
gs  = fig.add_gridspec(4, 3, hspace=0.42, wspace=0.32, width_ratios=[1.0, 2.2, 1.0])

hours_arr   = history[0]['network'].snapshots.hour
line_names  = list(history[0]['network'].lines.index)
n_lines     = len(line_names)
short_labels= [n.replace("Line_","").replace("_"," ") for n in line_names]

BUS_POS = {
    'Martin':(-122.4159,37.7089), 'Embarcadero':(-122.3892,37.7953),
    'Potrero_Switchyard':(-122.3898,37.7588), 'Mission':(-122.4102,37.7775),
    'Larkin':(-122.4194,37.7825), 'Bayshore':(-122.4035,37.7420),
    'HuntersPoint':(-122.3778,37.7306),
}
BUS_VNOMS = {
    'Martin':230,'Embarcadero':230,'Potrero_Switchyard':230,
    'Mission':115,'Larkin':115,'Bayshore':115,'HuntersPoint':115,
}

def loading_color(pct):
    if pct > 85: return '#d62728'
    if pct > 70: return '#ff7f0e'
    if pct > 50: return '#bcbd22'
    return '#2ca02c'

# =====================================================================
# Plot 1 — Expansion results bar chart
# =====================================================================
ax1 = fig.add_subplot(gs[0, 0])  # move expansion bar to top-left (single column)

depot_names_exp = list(EXISTING_DEPOTS.keys())
current_sizes = [expansion_results[n]['current'] for n in depot_names_exp]
expansions = [expansion_results[n]['expansion'] for n in depot_names_exp]

x_pos = np.arange(len(depot_names_exp))
w = 0.4

bars1 = ax1.bar(x_pos - w/2, current_sizes, w, label='Current', color='#1f77b4', alpha=0.8)
bars2 = ax1.bar(x_pos + w/2, expansions, w, label='Expansion', color='#2ca02c', alpha=0.8)

# Add value labels
for bar in bars1:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h, f'{int(h)}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar in bars2:
    h = bar.get_height()
    if h > 0:
        ax1.text(bar.get_x() + bar.get_width()/2, h, f'+{int(h)}',
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2ca02c')

ax1.set_xticks(x_pos)
ax1.set_xticklabels([n.replace('_',' ') for n in depot_names_exp], fontsize=10)
ax1.set_ylabel("Vehicles", fontsize=11)
ax1.set_title("Phase 1: Existing Depot Expansion Capacity", fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.25, axis='y')

# =====================================================================
# Plot 2 — New candidate ranking
# =====================================================================
ax2 = fig.add_subplot(gs[0, 2])

rank = {}
for h in history[1:]:
    if h['placed']:
        rank[h['placed']] = h['max_loading']
for c, ml in trial_loadings.items():
    rank[c] = ml

if rank:
    sorted_rank = sorted(rank.items(), key=lambda x: x[1])
    rnames = [x[0] for x in sorted_rank]
    rvals  = [x[1] for x in sorted_rank]
    rcolors= ['#2ca02c' if n in placed_new else '#d62728' for n in rnames]

    ax2.barh(range(len(rnames)), rvals, color=rcolors, ec='black', alpha=0.8, height=0.65)
    ax2.axvline(OVERLOAD_THRESHOLD, color='#d62728', ls='--', lw=2)
    ax2.set_yticks(range(len(rnames)))
    ax2.set_yticklabels([n.replace('_',' ') for n in rnames], fontsize=8.5)
    ax2.set_xlabel("Max line loading (%)", fontsize=10)
    ax2.set_title("Phase 2: New Depot Status", fontsize=11, fontweight='bold')
    legend_els = [Patch(fc='#2ca02c', ec='black', label='Placed'),
                  Patch(fc='#d62728', ec='black', label='Rejected')]
    ax2.legend(handles=legend_els, fontsize=8)
    ax2.grid(True, alpha=0.25, axis='x')
else:
    ax2.text(0.5, 0.5, 'No new depots\nevaluated', transform=ax2.transAxes,
            ha='center', va='center', fontsize=12)
    ax2.axis('off')

# =====================================================================
# Plot 3 — Network map (final state)
# =====================================================================
ax3 = fig.add_subplot(gs[0:2, 1])  # place network map centered at the top, spanning rows 0-1 in center column

final_ll   = history[-1]['line_loadings']
max_ll     = final_ll.max(axis=0)
final_net  = history[-1]['network']
final_config = history[-1]['config']

# Draw lines
for ln in line_names:
    b0 = final_net.lines.loc[ln, 'bus0']
    b1 = final_net.lines.loc[ln, 'bus1']
    x0, y0 = BUS_POS[b0]
    x1, y1 = BUS_POS[b1]
    pct    = max_ll[ln]
    col    = loading_color(pct)
    lw     = 5 if pct > 85 else 3.5 if pct > 70 else 2.5

    ax3.plot([x0,x1],[y0,y1], color=col, lw=lw, alpha=0.75, solid_capstyle='round', zorder=1)
    mx, my = (x0+x1)/2, (y0+y1)/2
    ax3.text(mx, my, f"{pct:.0f}%", fontsize=7, ha='center', va='center', zorder=4,
             bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=col, alpha=0.9, lw=1.2))

# Draw substations
LABEL_OFFSETS = {
    'Martin':(-18,-14), 'Embarcadero':(6,6), 'Potrero_Switchyard':(6,-12),
    'Mission':(-60,4), 'Larkin':(-55,-12), 'Bayshore':(-55,-10), 'HuntersPoint':(6,-12),
}

for bn, (bx, by) in BUS_POS.items():
    depots_here = [d for d in final_config.keys() if ALL_DEPOTS[d]['bus'] == bn]
    n_ev = sum(final_config[d] for d in depots_here)
    v    = BUS_VNOMS[bn]
    
    # Count existing vs new
    n_existing = sum(1 for d in depots_here if d in EXISTING_DEPOTS)
    n_new = len(depots_here) - n_existing

    if depots_here:
        # Color code: purple for existing, blue for new
        color = '#9467bd' if n_existing > 0 else '#17becf'
        ax3.scatter(bx, by, s=700, c=color, ec='gold', lw=3, zorder=5, marker='s')
        lbl = f"{bn}\n{n_ev} EVs"
        fc  = '#e6ccff'
    else:
        col = '#d62728' if v >= 230 else '#1f77b4'
        ax3.scatter(bx, by, s=350, c=col, ec='black', lw=2, zorder=5)
        lbl = bn
        fc  = '#ffffcc'

    ox, oy = LABEL_OFFSETS.get(bn, (6,6))
    ax3.annotate(lbl, (bx, by), xytext=(ox, oy), textcoords='offset points',
                 fontsize=7.5, fontweight='bold', ha='left',
                 bbox=dict(boxstyle='round,pad=0.3', fc=fc, ec='grey', alpha=0.92),
                 arrowprops=dict(arrowstyle='->', color='grey', lw=0.8) if (ox<-30 or ox>30) else None)

legend_els = [
    Line2D([0],[0], color='#2ca02c', lw=3, label='Line < 50 %'),
    Line2D([0],[0], color='#bcbd22', lw=3, label='Line 50-70 %'),
    Line2D([0],[0], color='#ff7f0e', lw=3.5, label='Line 70-85 %'),
    Line2D([0],[0], color='#d62728', lw=5,   label='Line > 85 %'),
    Line2D([0],[0], marker='s', color='w', markerfacecolor='#9467bd',
           markeredgecolor='gold', markersize=10, label='Expanded existing', linestyle='None'),
    Line2D([0],[0], marker='s', color='w', markerfacecolor='#17becf',
           markeredgecolor='gold', markersize=10, label='New depot', linestyle='None'),
]
ax3.legend(handles=legend_els, loc='lower left', fontsize=7.5, framealpha=0.95)
ax3.set_title(f"Network — {len(EXISTING_DEPOTS)} Expanded + {len(placed_new)} New Depots ({total_final_vehicles} vehicles)", 
              fontsize=13, fontweight='bold')
ax3.set_xlabel("Longitude", fontsize=9)
ax3.set_ylabel("Latitude", fontsize=9)
ax3.set_aspect('auto', adjustable='box')
ax3.grid(True, alpha=0.15)

# =====================================================================
# Plot 4 — Remaining headroom
# =====================================================================
ax4 = fig.add_subplot(gs[1, 2])
headroom = OVERLOAD_THRESHOLD - max_ll
hcolors  = ['#2ca02c' if v>12 else '#ff7f0e' if v>4 else '#d62728' for v in headroom]

ax4.barh(range(n_lines), headroom, color=hcolors, ec='black', alpha=0.8, height=0.65)
ax4.axvline(0, color='#d62728', lw=2)
ax4.set_yticks(range(n_lines))
ax4.set_yticklabels(short_labels, fontsize=7.5)
ax4.set_xlabel(f"Headroom to {OVERLOAD_THRESHOLD} % (%)", fontsize=10)
ax4.set_title("Remaining Line Headroom (Final)", fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.25, axis='x')

# =====================================================================
# Plot 5 & 6 — Loading heatmaps (baseline vs final)
# =====================================================================
# Baseline = current depots (before expansion)
net_baseline, _, _, ll_baseline = solve(base_config)

for col_idx, (title, ll_data) in enumerate([("Current Depots (baseline)", ll_baseline), 
                                              (f"Final ({total_final_vehicles} vehicles)", final_ll)]):
    ax = fig.add_subplot(gs[2, col_idx])
    data = ll_data.values.T

    im = ax.imshow(data, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=100, interpolation='nearest')
    ax.set_yticks(range(n_lines))
    ax.set_yticklabels(short_labels, fontsize=7)
    ax.set_xlabel("Hour", fontsize=9)
    ax.set_title(f"Line Loading Heatmap — {title}", fontsize=11, fontweight='bold')
    ax.set_xticks(range(0,24,3))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0,24,3)], fontsize=7)
    plt.colorbar(im, ax=ax, label='Loading %', shrink=0.85)

# =====================================================================
# Plot 8 — Fleet size progression
# =====================================================================
ax8 = fig.add_subplot(gs[3, 0])

# Show progression: current → expanded → final
stages = ['Current\nDepots', 'After\nExpansion', 'Final\n(+New Depots)']
vehicles = [total_current, total_expanded, total_final_vehicles]
colors_prog = ['#1f77b4', '#2ca02c', '#ff7f0e']

bars = ax8.bar(range(3), vehicles, color=colors_prog, alpha=0.8, edgecolor='black', linewidth=1.5)
for bar, v in zip(bars, vehicles):
    h = bar.get_height()
    ax8.text(bar.get_x() + bar.get_width()/2, h, f'{v}\nvehicles',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax8.set_xticks(range(3))
ax8.set_xticklabels(stages, fontsize=10)
ax8.set_ylabel("Total Fleet Size", fontsize=11)
ax8.set_title("Fleet Growth: Current → Expanded → Final", fontsize=11, fontweight='bold')
ax8.grid(True, alpha=0.25, axis='y')

# =====================================================================
# Plot 9 — Charger utilization (all depots in final config)
# =====================================================================
ax9 = fig.add_subplot(gs[3, 1])

final_net = history[-1]['network']
depot_names_all = list(final_config.keys())
n_depots = len(depot_names_all)
COLORS = plt.cm.tab10(np.linspace(0, 1, max(n_depots, 1)))

for i, depot_name in enumerate(depot_names_all):
    charger_name = f"{depot_name}_charger"
    if charger_name in final_net.links.index:
        power = final_net.links_t.p0[charger_name]
        cap   = final_net.links.loc[charger_name, 'p_nom']
        util  = (power / cap) * 100
        
        # Style: existing depots solid, new depots dashed
        style = '-' if depot_name in EXISTING_DEPOTS else '--'
        lw = 2.5 if depot_name in EXISTING_DEPOTS else 2
        
        ax9.plot(hours_arr, util, lw=lw, ls=style, label=depot_name.replace('_',' '),
                 color=COLORS[i], marker='o', ms=2.5)

ax9.axhline(100, color='#d62728', ls='--', lw=1.5, alpha=0.6, label='Full capacity')
ax9.set_title("Charger Utilization — All Depots", fontsize=11, fontweight='bold')
ax9.set_xlabel("Hour", fontsize=10)
ax9.set_ylabel("Utilization (%)", fontsize=10)
ax9.legend(fontsize=6.5, loc='upper right', ncol=2)
ax9.grid(True, alpha=0.25)
ax9.set_xlim(0, 23)
ax9.set_ylim(0, 110)

# =====================================================================
# Plot 10 — Summary text
# =====================================================================
ax10 = fig.add_subplot(gs[2, 2])
ax10.axis('off')

# Tightest line
tightest_line = max_ll.idxmax()
tightest_val  = max_ll.max()

summary = (
    f"FINAL CONFIGURATION\n"
    f"{'━' * 35}\n\n"
    f"EXISTING DEPOTS (expanded):\n"
)
for name in EXISTING_DEPOTS.keys():
    current = base_config[name]
    final = expanded_config[name]
    expansion = final - current
    summary += f"  {name:16s} {final:3d} (+{expansion:2d})\n"

summary += f"\nNEW DEPOTS (greenfield):\n"
if placed_new:
    for name in placed_new:
        summary += f"  {name:16s} {STANDARD_DEPOT_SIZE:3d}\n"
else:
    summary += f"  (none placed)\n"

summary += (
    f"\n{'━' * 35}\n"
    f"TOTAL FLEET:\n"
    f"  Depots:    {total_depots}\n"
    f"  Vehicles:  {total_final_vehicles}\n"
    f"  Growth:    +{total_final_vehicles - total_current} from current\n"
    f"\nGRID IMPACT:\n"
    f"  Max loading:    {history[-1]['max_loading']:.1f} %\n"
    f"  Tightest line:  {tightest_line}\n"
    f"                  ({tightest_val:.1f} %)\n"
    f"  Operating cost: ${history[-1]['objective']:,.0f}\n"
    f"\n{'━' * 35}\n"
    f"REJECTED NEW SITES:\n"
)
if remaining_rejected:
    for cand in sorted(remaining_rejected):
        summary += f"  ✗ {cand:14s} {trial_loadings[cand]:.1f} %\n"
else:
    summary += f"  (all candidates fit)\n"

ax10.text(0.04, 0.96, summary, transform=ax10.transAxes,
          fontsize=8, verticalalignment='top', fontfamily='monospace',
          bbox=dict(boxstyle='round', facecolor='#e6f2ff', ec='grey', alpha=0.95))

plt.tight_layout()
plt.savefig('sf_grid_waymo_expansion_placement.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: sf_grid_waymo_expansion_placement.png")

print("=" * 80)