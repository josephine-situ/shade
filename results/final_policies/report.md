# SHADE: Exported Policies
**Run:** runs/evolve_20260823T045409Z  
**Exported:** 2026-08-23 23:44 UTC  
**Policies:** 3

## synergy-triple-shade corridors v1
**ID:** gen47_6daffb24  
**Generation:** 47  
**Model:** claude-sonnet-4-6  

**Description:** Maximise heat_relief_c via three-tier shade deployment with synergy-based 

### Stakeholder selection
**Selected for:** Boston Public Health Commission  
**Rationale:** Maximizes the number of residents moved below heat stress thresholds while delivering strong equity — vulnerable populations are the ones most at risk of heat-related illness and death. Highest equity ratio (1.68) among top-relief policies ensures interventions reach the neighborhoods that need them most.

| Metric | Weight |
|---|---|
| Heat relief (°C) | 0.30 |
| Equity ratio | 0.25 |
| Access gain (pp) | 0.35 |
| Cost efficiency (person·°C/$100k) | 0.05 |
| Greening co-benefit (%) | 0.05 |

### Objectives (evolution scoring)
| Metric | Value |
|---|---|
| Heat relief (°C) | 0.0323
| Equity ratio | 1.7236
| Access gain (pp) | 0.4583
| Cost efficiency (person·°C/$100k) | 49.1500
| Greening co-benefit (%) | 0.2213

### Objectives (full validation: 20 AOIs x 4 scenarios)
| Metric | Value |
|---|---|
| Heat relief (°C) | 0.0305
| Equity ratio | 1.6767
| Access gain (pp) | 0.1136
| Cost efficiency (person·°C/$100k) | 45.5000
| Greening co-benefit (%) | 0.2027

### Lineage
**gen 47** (gen47_6daffb24)  fitness=0.0323  "synergy-triple-shade corridors v1"
    _Maximise heat_relief_c via three-tier shade deployment with synergy-based _
    + inspiration: gen 33 (gen33_1022f141)  fitness=0.0313  "synergy-shade heat-equity corridors v2"
    + inspiration: gen 2 (gen02_5c645778)  fitness=0.0260  "tight-tree corridors with canopy infill"
  └── **gen 29** (gen29_3360b7e4)  fitness=0.0260  "dense-shade corridors v2"
      _Maximise heat_relief_c via aggressive shade deployment: _
      + inspiration: gen 2 (gen02_5c645778)  fitness=0.0260  "tight-tree corridors with canopy infill"
      + inspiration: gen 28 (gen28_9d08d3cb)  fitness=0.0230  "equity-heat medium tree corridors v2"
    └── **gen 21** (gen21_60cc7c52)  fitness=0.0293  "triple-action heat-equity corridors v1"
        _Maximise heat_relief_c (population-weighted UTCI drop) with a triple-action _
        + inspiration: gen 9 (gen09_c7e2c15b)  fitness=0.0290  "dual-tree shade corridors v2"
        + inspiration: gen 3 (gen03_1dd7db33)  fitness=0.0270  "heat-first medium trees with canopy infill v2"
      └── **gen 2** (gen02_5c645778)  fitness=0.0260  "tight-tree corridors with canopy infill"
          _Build a heat × heat-hours × vulnerability × population priority surface _
          + inspiration: gen 1 (gen01_f854af82)  fitness=0.0253  "dense-small-tree equity corridors"
        └── **gen 0** (gen00_seed)  fitness=0.0293  "hot-corridor trees, then canopies"
            _Rank pedestrian space by heat, vulnerability and footfall; plant medium _

## synergy-equity-greening dense v2
**ID:** gen73_10d1ce72  
**Generation:** 73  
**Model:** claude-sonnet-4-6  

**Description:** Cell (1,0,0,1): equity_ratio>=1, cobenefit_greened_pct>=0.1976, 

### Stakeholder selection
**Selected for:** Environmental Justice Advocate (e.g. GreenRoots, ACE)  
**Rationale:** Highest greening co-benefit (0.29) combined with strong equity (1.52). Trees and green space are permanent community assets that deliver benefits beyond heat — air quality, stormwater management, mental health, property stabilization. For communities that have been disproportionately burdened by environmental harm, temporary fixes are not enough; equitable, lasting green infrastructure is the priority.

| Metric | Weight |
|---|---|
| Heat relief (°C) | 0.20 |
| Equity ratio | 0.40 |
| Access gain (pp) | 0.10 |
| Cost efficiency (person·°C/$100k) | 0.05 |
| Greening co-benefit (%) | 0.25 |

### Objectives (evolution scoring)
| Metric | Value |
|---|---|
| Heat relief (°C) | 0.0277
| Equity ratio | 1.7028
| Access gain (pp) | 0.1782
| Cost efficiency (person·°C/$100k) | 42.0200
| Greening co-benefit (%) | 0.3153

### Objectives (full validation: 20 AOIs x 4 scenarios)
| Metric | Value |
|---|---|
| Heat relief (°C) | 0.0292
| Equity ratio | 1.5184
| Access gain (pp) | 0.1065
| Cost efficiency (person·°C/$100k) | 43.8100
| Greening co-benefit (%) | 0.2910

### Lineage
**gen 73** (gen73_10d1ce72)  fitness=0.0277  "synergy-equity-greening dense v2"
    _Cell (1,0,0,1): equity_ratio>=1, cobenefit_greened_pct>=0.1976, _
    + inspiration: gen 60 (gen60_980daac3)  fitness=0.0342  "dense-synergy-corridors no-equity-boost v1"
    + inspiration: gen 25 (gen25_e257ea3f)  fitness=0.0303  "dense-triple-action heat-population v4"
  └── **gen 72** (gen72_a9355d37)  fitness=0.0260  "synergy-equity dense corridors v1"
      _Combines the high-fitness dense synergy approach (3m medium tree spacing, _
      + inspiration: gen 50 (gen50_2931c524)  fitness=0.0263  "synergy-corridor heat-equity v1"
      + inspiration: gen 60 (gen60_980daac3)  fitness=0.0342  "dense-synergy-corridors no-equity-boost v1"
    └── **gen 1** (gen01_f854af82)  fitness=0.0253  "dense-small-tree equity corridors"
        _Build a heat × vulnerability × population priority surface, then fill _
      └── **gen 0** (gen00_seed)  fitness=0.0293  "hot-corridor trees, then canopies"
          _Rank pedestrian space by heat, vulnerability and footfall; plant medium _

## dense-synergy-tight-trees no-equity v2
**ID:** gen99_33e4e664  
**Generation:** 99  
**Model:** claude-sonnet-4-6  

**Description:** Maximise heat_relief_c in cell (0,1,1,0): low equity_ratio, high 

### Stakeholder selection
**Selected for:** City Chief Financial Officer  
**Rationale:** Maximum heat relief per dollar spent (50.58 person-degC per $100k), the highest absolute relief (0.0328°C), and the most residents moved below the stress threshold (0.1273). This is the policy that makes the strongest case in a budget hearing: every dollar produces measurable, defensible impact. Equity is lower (1.08) but still above 1.0, meaning vulnerable communities still benefit proportionally.

| Metric | Weight |
|---|---|
| Heat relief (°C) | 0.30 |
| Equity ratio | 0.10 |
| Access gain (pp) | 0.20 |
| Cost efficiency (person·°C/$100k) | 0.35 |
| Greening co-benefit (%) | 0.05 |

### Objectives (evolution scoring)
| Metric | Value |
|---|---|
| Heat relief (°C) | 0.0343
| Equity ratio | 1.0398
| Access gain (pp) | 0.2745
| Cost efficiency (person·°C/$100k) | 54.2900
| Greening co-benefit (%) | 0.2173

### Objectives (full validation: 20 AOIs x 4 scenarios)
| Metric | Value |
|---|---|
| Heat relief (°C) | 0.0328
| Equity ratio | 1.0754
| Access gain (pp) | 0.1273
| Cost efficiency (person·°C/$100k) | 50.5800
| Greening co-benefit (%) | 0.2019

### Lineage
**gen 99** (gen99_33e4e664)  fitness=0.0343  "dense-synergy-tight-trees no-equity v2"
    _Maximise heat_relief_c in cell (0,1,1,0): low equity_ratio, high _
    + inspiration: gen 50 (gen50_2931c524)  fitness=0.0263  "synergy-corridor heat-equity v1"
    + inspiration: gen 60 (gen60_980daac3)  fitness=0.0342  "dense-synergy-corridors no-equity-boost v1"
  └── **gen 13** (gen13_b4c81ffa)  fitness=0.0297  "dense-corridor dual-tree canopy v3"
      _Maximise population-weighted UTCI relief via three-phase shading: _
      + inspiration: gen 5 (gen05_ef962afb)  fitness=0.0240  "population-heat max shade v2"
      + inspiration: gen 0 (gen00_seed)  fitness=0.0293  "hot-corridor trees, then canopies"
    └── **gen 3** (gen03_1dd7db33)  fitness=0.0270  "heat-first medium trees with canopy infill v2"
        _Maximise UTCI relief by aggressively targeting the hottest pedestrian _
        + inspiration: gen 0 (gen00_seed)  fitness=0.0293  "hot-corridor trees, then canopies"
        + inspiration: gen 1 (gen01_f854af82)  fitness=0.0253  "dense-small-tree equity corridors"
      └── **gen 2** (gen02_5c645778)  fitness=0.0260  "tight-tree corridors with canopy infill"
          _Build a heat × heat-hours × vulnerability × population priority surface _
          + inspiration: gen 1 (gen01_f854af82)  fitness=0.0253  "dense-small-tree equity corridors"
        └── **gen 0** (gen00_seed)  fitness=0.0293  "hot-corridor trees, then canopies"
            _Rank pedestrian space by heat, vulnerability and footfall; plant medium _
