# San Francisco Grid SCLOPF Project

A PyPSA model for analyzing the San Francisco electrical grid using Security-Constrained Linear Optimal Power Flow (SCLOPF), developed after the December 2024 Mission substation fire that left 130,000 customers without power.

## Background

On December 21, 2024, a fire at PG&E's Mission substation (8th and Mission Streets) caused widespread outages affecting approximately 130,000 customers across San Francisco. The incident highlighted the need for better contingency planning and security-constrained optimization of the grid.

## SCLOPF

Security-Constrained Linear Optimal Power Flow ensures that:
- The power system operates optimally under normal conditions
- No line becomes overloaded if any single branch fails (N-1 security)
- Generation is pre-positioned to handle contingencies
- The system can survive equipment failures without cascading blackouts

This is achieved using Branch Outage Distribution Factors (BODF) to model how power flows redistribute when a line fails.

## Installation

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyPSA and dependencies
pip install pypsa pandas numpy matplotlib

# Install optimization solver
pip install highspy  # Or use: conda install -c conda-forge coincbc
```

## Project Structure

```
sf-pypsa-sclopf/
├── sf_grid_model.py          # Main model script
├── README.md                 # This file
├── data/                     # Data directory 
│   ├── load_profiles.csv    # Historical load data, to fill
│   └── generation_data.csv  # Generation capacity data, to fill
└── sf_grid_sclopf_analysis.png  # Visualizations
    
```

## Model Components

### Buses (Substations)
Pulled data infrastructure from https://cecgis-caenergy.opendata.arcgis.com/search?categories=%252Fcategories%252Fenergy%2520infrastructure
See https://openinframap.org/#11.14/37.7225/-122.3614 for a visualization of SF substations. 
Based on OpenInfraMap and CEC GIS data showing actual substations:
- Larkin 115kV Substation
- Mission 115kV Substation  
- Embarcadero 230kV Substation
- Potrero Switchyard 230kV (next to Potrero Converter Station 200kV DC)
- Bayshore 115kV Substation
- Hunters Point 115kV Substation
- Martin 230kV Substation

### Generators
1. **Hetch Hetchy** (400 MW) - SFPUC hydropower from Yosemite
2. **External Grid Martin** (1,200 MW) - PG&E broader system connection
3. **Trans Bay Cable** (400 MW) - Underwater cable from Pittsburg
4. **Backup Generators** (50 MW) - Emergency diesel generation

### Transmission Lines
11 transmission lines connecting substations with estimated:
- Impedances (resistance and reactance)
- Capacity ratings (MVA)
- Geographic distances (used straight-line distance x1.3 routing factor)

## Running the Model

### Basic Usage

```bash
python sf_grid_model.py
```

This will:
1. Build the SF grid network
2. Run standard Linear OPF (LOPF)
3. Run Security-Constrained LOPF (SCLOPF)
4. Generate comparison analysis
5. Create visualization plots

### Expected Output

```
======================================================================
SAN FRANCISCO GRID SECURITY-CONSTRAINED OPTIMAL POWER FLOW
======================================================================

Network: 9 buses, 15 lines
Generators: 4
Total load: 650 MW (base)

1. Running standard Linear Optimal Power Flow (LOPF)...
   Objective value: $XXX,XXX.XX
   Generation dispatch:
     HetchHetchy: XXX.X MW (avg)
     ExternalGrid: XXX.X MW (avg)
     ...

2. Running Security-Constrained LOPF (SCLOPF)...
   Analyzing 5 critical line outage scenarios:
     - Line_0_Martin_Mission
     - Line_3_Mission_Larkin
     - Line_4_Mission_Embarcadero
     - Line_5_Mission_GlenPark
```

## References

- [PyPSA Documentation](https://pypsa.readthedocs.io/)
- [SCLOPF Example](https://pypsa.readthedocs.io/en/latest/examples/scigrid-sclopf.html)
- [PG&E December 2024 Outage Report](https://www.pge.com/en/newsroom/currents/safety/pg-e-responding-to-power-outage-in-san-francisco-.html)
- [SF Chronicle Coverage](https://www.sfchronicle.com/sf/article/pg-e-outage-40-000-customers-without-power-21254326.php)
- [Trans-Bay Cable]( https://en.wikipedia.org/wiki/Trans_Bay_Cable)
- [Transmission line data] (https://cecgis-caenergy.opendata.arcgis.com/datasets/CAEnergy::california-electric-transmission-lines-1/explore)

## Potential Enhancements

To improve this model:
1. Add actual SF grid topology data
2. Calibrate parameters with PG&E technical specs and read load data
### Adding Real Load Data
```python
# Replace synthetic load profile with PG&E data
import pandas as pd
load_data = pd.read_csv('data/load_profiles.csv', index_col=0, parse_dates=True)
for bus, load_mw in loads.items():
    network.add("Load", f"Load_{bus}", 
                bus=bus,
                p_set=load_data[bus])
```


3. Validate against real outage scenarios
4. Include more substations and detail
5. Include storage units (batteries) or other generation
### Adding Battery Storage
```python
# Model battery energy storage systems
network.add("StorageUnit", "SFBattery",
            bus="Mission",
            p_nom=100,      # MW charging/discharging capacity
            max_hours=4,    # 400 MWh energy capacity
            efficiency_store=0.95,
            efficiency_dispatch=0.95,
            cyclic_state_of_charge=True)
```

### Adding Solar Generation
```python
# Model rooftop solar variability
solar_profile = pd.read_csv('data/solar_irradiance.csv')
network.add("Generator", "DistributedSolar",
            bus="Richmond",
            p_nom=200,
            marginal_cost=0,
            p_max_pu=solar_profile['normalized_output'])
```
6. Create a price responsive model.
7. Wildfire scenarios relevant to CA

