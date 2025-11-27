# HRC_RTRC

This repository contains the full implementation of **Harvested Reservoir Computing from Road Traffic Dynamics (HRC / RTRC)**.  
It includes numerical simulations, scaled traffic experiments, and scripts used to generate all figures in the paper.

---

## Repository Structure

| Folder | Description |
|--------|-------------|
| **grid_road_traffic_sim/** | Numerical simulations for grid road networks |
| **scaled_traffic_experiment/** | Codes for scaled traffic experiments (physical miniature vehicles) |
| **scaled_traffic_experiment_sample_video/** | Sample videos from the scaled experiments |

---

## How to Reproduce the Figures

Below is a list of scripts used to generate each figure in the paper.

### **Figure 3**
```bash
python scaled_traffic_RTRC.py
```

### **Figure 6 (a)–(d)**
```bash
python scaled_traffic_RTRC.py
```

### **Figure 7 (a), (b)**
```bash
python scaled_traffic_RTRC.py
```

### **Figure 8 (a), (b)**
```bash
python result_pred.py
```

### **Figure 9**
1. First run the simulation with
```
grid_road_traffic_sim/detail_analysis/asynchronous_traffic_signal/road_traffic.py
```
2. Then:
```
python grid_road_traffic_sim/detail_analysis/asynchronous_traffic_signal/result_pred.py
```

### **Figure 10**
```bash
python calc_FD.py
```

### **Figure 11**
```bash
python result_FD.py
```

### **Figure 12**
```bash
python result_FD.py
```

### **Figure 13**
```bash
python result_all.py
```

### **Figure 14**
```bash
python result_all.py
```

### **Figure 15**
```bash
python result_all.py
```

### **Supplementary Figures B3, B4, B5**
```bash
python calc_MC.py
```

---

## Additional Analyses Used in the Paper

### **Realistic routing & open-boundary conditions (Section: Results)**
The analysis for realistic routing is implemented in:

```
grid_road_traffic_sim/detail_analysis/6times6_grid/
```

### **IPC analysis (Section: Discussion)**
Scripts used to compute IPC (Information Processing Capacity):

```
grid_road_traffic_sim/detail_analysis/calc_IPC/
```

---

## Notes
- Some figures require prior generation of simulation result files.
- Folder paths are relative to this repository (no absolute user-specific paths).

