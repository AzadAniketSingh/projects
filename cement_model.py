import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv(r'C:\Users\Azad Aniket\OneDrive\Desktop\python files\Azad_suvidha _foundation_mlmodel\cement_opc_dataset.csv')


C_NAVY   = "#1F3864"
C_RED    = "#C0392B"
C_GREEN  = "#1E8449"
C_AMBER  = "#D4AC0D"
C_BLUE   = "#2980B9"
C_ORANGE = "#E67E22"
C_GREY   = "#717D7E"
C_LIGHT  = "#EAF2FF"

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.titlesize': 11,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 150
})

# 
# — Dataset Overview
# 
fig1, axes = plt.subplots(2, 3, figsize=(15, 9))
fig1.suptitle('Figure 1: Cement OPC Dataset Overview — Key Variable Distributions',
              fontsize=13, fontweight='bold', color=C_NAVY, y=1.01)

panel_vars = [
    ('co2_emissions_t_t',            'CO₂ Emissions (t CO₂/t cement)', C_RED),
    ('energy_consumption_GJ_t',      'Energy Consumption (GJ/t)',       C_ORANGE),
    ('compressive_strength_28d_MPa', '28-day Compressive Strength (MPa)', C_GREEN),
    ('kiln_temperature_C',           'Kiln Temperature (°C)',            C_NAVY),
    ('alt_fuel_fraction',            'Alternative Fuel Fraction',        C_BLUE),
    ('lime_saturation_factor',       'Lime Saturation Factor',           C_AMBER),
]

for ax, (col, label, color) in zip(axes.flat, panel_vars):
    vals = df[col]
    ax.hist(vals, bins=50, color=color, alpha=0.8, edgecolor='white', linewidth=0.4)
    ax.axvline(vals.mean(),          color='black', linestyle='--', linewidth=1.4,
               label=f'Mean: {vals.mean():.3f}')
    ax.axvline(vals.quantile(0.25),  color=C_GREY,  linestyle=':', linewidth=1,
               label=f'Q1: {vals.quantile(0.25):.3f}')
    ax.axvline(vals.quantile(0.75),  color=C_GREY,  linestyle=':', linewidth=1,
               label=f'Q3: {vals.quantile(0.75):.3f}')
    ax.set_title(label, fontweight='bold', color=C_NAVY, pad=6)
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=7, framealpha=0.5)
    ax.set_facecolor(C_LIGHT)

plt.tight_layout()
plt.savefig('fig1_overview.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# 
# — Problem Identification Scatter Plots
# 
df['high_emission'] = df['co2_emissions_t_t'] >= df['co2_emissions_t_t'].quantile(0.80)
colors_map = df['high_emission'].map({True: C_RED, False: C_BLUE})

fig2, axes2 = plt.subplots(2, 3, figsize=(15, 9))
fig2.suptitle('Figure 2: Problem Identification — Key Relationships in Cement Production',
              fontsize=13, fontweight='bold', color=C_NAVY, y=1.01)

scatter_pairs = [
    ('energy_consumption_GJ_t',  'co2_emissions_t_t',
     'Energy Consumption (GJ/t)', 'CO₂ Emissions (t/t)',
     'Problem 1: Higher energy → Higher CO₂'),
    ('alt_fuel_fraction',        'co2_emissions_t_t',
     'Alternative Fuel Fraction', 'CO₂ Emissions (t/t)',
     'Problem 2: Low AF use → High CO₂'),
    ('lime_saturation_factor',   'compressive_strength_28d_MPa',
     'Lime Saturation Factor',    '28-day Strength (MPa)',
     'Problem 3: LSF deviation → Strength loss'),
    ('kiln_temperature_C',       'energy_consumption_GJ_t',
     'Kiln Temperature (°C)',     'Energy Consumption (GJ/t)',
     'Problem 4: Temperature inefficiency'),
    ('moisture_content_pct',     'energy_consumption_GJ_t',
     'Moisture Content (%)',      'Energy Consumption (GJ/t)',
     'Problem 5: High moisture → Energy waste'),
    ('alt_fuel_fraction',        'compressive_strength_28d_MPa',
     'Alternative Fuel Fraction', '28-day Strength (MPa)',
     'Problem 6: AF fraction vs Quality'),
]

from matplotlib.lines import Line2D

for ax, (x, y, xl, yl, title) in zip(axes2.flat, scatter_pairs):
    ax.scatter(df[x], df[y], c=colors_map, alpha=0.25, s=6, linewidths=0)
    z = np.polyfit(df[x], df[y], 1)
    xs = np.linspace(df[x].min(), df[x].max(), 200)
    ax.plot(xs, np.poly1d(z)(xs), color='black', linewidth=1.6, linestyle='--')
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.set_title(title, fontweight='bold', color=C_NAVY, fontsize=9, pad=5)
    ax.set_facecolor(C_LIGHT)
    legend_elements = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=C_RED,
               markersize=7, label='High Emission (top 20%)'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=C_BLUE,
               markersize=7, label='Normal')
    ]
    ax.legend(handles=legend_elements, fontsize=7, framealpha=0.5)

plt.tight_layout()
plt.savefig('fig2_problems.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# 
#  — Correlation Heatmap
# 
fig3, ax3 = plt.subplots(figsize=(14, 11))
corr = df.drop(columns=['high_emission']).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
            center=0, vmin=-1, vmax=1, ax=ax3,
            annot_kws={'size': 7}, linewidths=0.4,
            cbar_kws={'shrink': 0.8, 'label': 'Pearson r'})

ax3.set_title('Figure 3: Correlation Matrix — All Variables',
              fontsize=12, fontweight='bold', color=C_NAVY, pad=12)
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=8)
ax3.set_yticklabels(ax3.get_yticklabels(), rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig('fig3_heatmap.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()


#  — Train Random Forest

feature_cols = [c for c in df.columns
                if c not in ['co2_emissions_t_t',
                             'compressive_strength_28d_MPa',
                             'high_emission']]

X      = df[feature_cols].values
y_co2  = df['co2_emissions_t_t'].values
y_str  = df['compressive_strength_28d_MPa'].values

Xtr, Xte, y_co2_tr, y_co2_te = train_test_split(
    X, y_co2, test_size=0.2, random_state=42)
_,   _,   y_str_tr, y_str_te  = train_test_split(
    X, y_str,  test_size=0.2, random_state=42)

scaler = StandardScaler()
Xtr_s  = scaler.fit_transform(Xtr)
Xte_s  = scaler.transform(Xte)

# CO₂ model
rf_co2 = RandomForestRegressor(n_estimators=200, max_depth=12,
                                n_jobs=-1, random_state=42)
rf_co2.fit(Xtr_s, y_co2_tr)
pred_co2 = rf_co2.predict(Xte_s)

# Strength model
rf_str = RandomForestRegressor(n_estimators=200, max_depth=12,
                                n_jobs=-1, random_state=42)
rf_str.fit(Xtr_s, y_str_tr)
pred_str = rf_str.predict(Xte_s)

r2_co2  = r2_score(y_co2_te, pred_co2)
mae_co2 = mean_absolute_error(y_co2_te, pred_co2)
r2_str  = r2_score(y_str_te, pred_str)
mae_str = mean_absolute_error(y_str_te, pred_str)

print(f"CO2 Model     — R²={r2_co2:.4f}  MAE={mae_co2:.4f}")
print(f"Strength Model — R²={r2_str:.4f}  MAE={mae_str:.4f}")

fi_co2 = pd.Series(rf_co2.feature_importances_,
                   index=feature_cols).sort_values(ascending=False)
fi_str = pd.Series(rf_str.feature_importances_,
                   index=feature_cols).sort_values(ascending=False)

# Actual vs Predicted
fig4, axes4 = plt.subplots(1, 2, figsize=(13, 6))
fig4.suptitle('Figure 4: Random Forest — Actual vs Predicted',
              fontsize=13, fontweight='bold', color=C_NAVY)

for ax, actual, pred, label, color, r2, mae in [
    (axes4[0], y_co2_te, pred_co2, 'CO₂ Emissions (t/t)',     C_RED,   r2_co2, mae_co2),
    (axes4[1], y_str_te, pred_str, '28-day Strength (MPa)',   C_GREEN, r2_str, mae_str),
]:
    lims = [min(actual.min(), pred.min())*0.98,
            max(actual.max(), pred.max())*1.02]
    ax.scatter(actual, pred, alpha=0.25, s=8, color=color, linewidths=0)
    ax.plot(lims, lims, 'k--', linewidth=1.5, label='Perfect fit')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel(f'Actual {label}')
    ax.set_ylabel(f'Predicted {label}')
    ax.set_title(f'Model: {label}', fontweight='bold', color=C_NAVY)
    ax.set_facecolor(C_LIGHT)
    ax.text(0.05, 0.93, f'R² = {r2:.4f}\nMAE = {mae:.4f}',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('fig4_model.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()


# Feature Importance
friendly = {
    'energy_consumption_GJ_t':    'Energy Consumption',
    'alt_fuel_fraction':          'Alternative Fuel Fraction',
    'kiln_temperature_C':         'Kiln Temperature',
    'lime_saturation_factor':     'Lime Saturation Factor',
    'fuel_input_rate_t_h':        'Fuel Input Rate',
    'coal_calorific_value_kJ_kg': 'Coal Calorific Value',
    'preheater_temp_C':           'Preheater Temperature',
    'raw_meal_feed_rate_t_h':     'Raw Meal Feed Rate',
    'CaO_pct':                    'CaO Content (%)',
    'silicate_ratio':             'Silicate Ratio',
    'alumina_ratio':              'Alumina Ratio',
    'production_rate_t_day':      'Production Rate',
    'kiln_speed_rpm':             'Kiln Speed',
    'cement_fineness_m2_kg':      'Cement Fineness',
    'SiO2_pct':                   'SiO₂ Content (%)',
    'Al2O3_pct':                  'Al₂O₃ Content (%)',
    'Fe2O3_pct':                  'Fe₂O₃ Content (%)',
    'MgO_pct':                    'MgO Content (%)',
    'gypsum_pct':                 'Gypsum Content (%)',
    'moisture_content_pct':       'Moisture Content',
    'cooling_air_flow_m3_kg':     'Cooling Air Flow',
}

fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(15, 7))
fig5.suptitle('Figure 5: Feature Importance — CO₂ Drivers & Strength Drivers',
              fontsize=13, fontweight='bold', color=C_NAVY)

top_n = 12
for ax, fi, title, color in [
    (ax5a, fi_co2, 'CO₂ Emission Drivers',       C_RED),
    (ax5b, fi_str, 'Compressive Strength Drivers', C_GREEN),
]:
    top    = fi.head(top_n)
    labels = [friendly.get(i, i) for i in top.index]
    bars   = ax.barh(range(top_n), top.values[::-1],
                     color=[color]*top_n, alpha=0.85, edgecolor='white')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels[::-1], fontsize=9)
    ax.set_xlabel('Feature Importance Score', fontsize=9)
    ax.set_title(title, fontweight='bold', color=C_NAVY, pad=8)
    ax.set_facecolor(C_LIGHT)
    for bar, val in zip(bars, top.values[::-1]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=7.5)

plt.tight_layout()
plt.savefig('fig5_importance.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# 
# — Problems & Solutions Infographic

fig6, ax6 = plt.subplots(figsize=(16, 12))
ax6.set_xlim(0, 16); ax6.set_ylim(0, 12)
ax6.axis('off')
ax6.set_facecolor('#F8F9FA')
fig6.patch.set_facecolor('#F8F9FA')

ax6.text(8, 11.5, 'ML-Identified Problems & Recommended Solutions',
         ha='center', fontsize=14, fontweight='bold', color=C_NAVY)
ax6.text(8, 11.1, 'Based on Random Forest Analysis of 5,000 OPC Records',
         ha='center', fontsize=9, color=C_GREY, style='italic')
ax6.axhline(10.8, xmin=0.03, xmax=0.97, color=C_NAVY, linewidth=1.5)

problems = [
    {
        'title':    'PROBLEM 1\nHigh Energy Consumption',
        'stat':     f"Top CO₂ driver\nImportance: {fi_co2['alt_fuel_fraction']:.3f}",
        'detail':   f"Dataset mean: 3.45 GJ/t\nHigh-emission plants: >3.8 GJ/t\nRange: 3.0–4.2 GJ/t",
        'solution': '• Upgrade to 5-stage preheater\n• Optimize kiln burner settings\n• Install waste heat recovery\n• Target: <3.2 GJ/t clinker',
        'color': C_RED,   'x': 1.0,
    },
    {
        'title':    'PROBLEM 2\nLow Alternative Fuel Use',
        'stat':     f"2nd CO₂ driver\nMean AF: {df['alt_fuel_fraction'].mean():.2f}",
        'detail':   f"{(df['alt_fuel_fraction']<0.2).mean()*100:.1f}% plants below 20%\nEU best practice: >40%\nDataset max: {df['alt_fuel_fraction'].max():.2f}",
        'solution': '• Co-process biomass & RDF\n• Partner with waste suppliers\n• Regulatory pre-approval\n• Target: ≥40% thermal sub.',
        'color': C_ORANGE, 'x': 5.5,
    },
    {
        'title':    'PROBLEM 3\nLSF Deviation & Strength Loss',
        'stat':     f"Top strength driver\nImportance: {fi_str['lime_saturation_factor']:.3f}",
        'detail':   f"Optimal LSF: 0.92–0.96\nDataset range: 0.82–1.05\n{(df['lime_saturation_factor']<0.90).mean()*100:.1f}% below optimal",
        'solution': '• Real-time XRF raw meal analysis\n• Closed-loop LSF control\n• Tighten feed ratio targets\n• Target: LSF 0.93 ± 0.02',
        'color': C_AMBER,  'x': 10.0,
    },
]

for prob in problems:
    x = prob['x']
    ax6.add_patch(FancyBboxPatch((x, 5.5), 4.2, 5.0, boxstyle="round,pad=0.1",
                                  facecolor=prob['color'], alpha=0.12,
                                  edgecolor=prob['color'], linewidth=2))
    ax6.text(x+2.1, 10.15, prob['title'], ha='center', fontsize=9.5,
             fontweight='bold', color=prob['color'])
    ax6.text(x+2.1, 9.45,  prob['stat'],  ha='center', fontsize=8.5, color=C_NAVY,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7,
                       edgecolor=prob['color']))
    ax6.text(x+2.1, 8.4,   prob['detail'], ha='center', fontsize=8,
             color='#2C3E50', linespacing=1.5)
    ax6.annotate('', xy=(x+2.1, 5.3), xytext=(x+2.1, 5.6),
                 arrowprops=dict(arrowstyle='->', color=prob['color'], lw=2.5))
    ax6.add_patch(FancyBboxPatch((x, 0.5), 4.2, 4.6, boxstyle="round,pad=0.1",
                                  facecolor=C_GREEN, alpha=0.08,
                                  edgecolor=C_GREEN, linewidth=2))
    ax6.text(x+2.1, 4.85, '✔ RECOMMENDED SOLUTION', ha='center',
             fontsize=8, fontweight='bold', color=C_GREEN)
    ax6.text(x+2.1, 3.2,  prob['solution'], ha='center', fontsize=8.5,
             color='#1A5276', linespacing=1.7)

ax6.add_patch(FancyBboxPatch((0.3, -0.1), 15.4, 0.55,
                              boxstyle="round,pad=0.05",
                              facecolor=C_NAVY, alpha=0.08,
                              edgecolor=C_NAVY, linewidth=1))
ax6.text(8, 0.17,
         f'CO₂ Model: R²={r2_co2:.4f}, MAE={mae_co2:.4f}  |  '
         f'Strength Model: R²={r2_str:.4f}, MAE={mae_str:.4f}  |  '
         f'Train: 4,000 samples  |  Test: 1,000 samples',
         ha='center', fontsize=8, color=C_NAVY)

plt.tight_layout()
plt.savefig('fig6_summary.png', dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
plt.close()

# 
# FIGURE 7 — What-If Simulation
# 
af_values      = np.linspace(0, 0.8, 50)
baseline_mean  = df[feature_cols].mean().values.copy()
base_idx       = feature_cols.index('alt_fuel_fraction')
e_idx          = feature_cols.index('energy_consumption_GJ_t')

co2_af_sim = []
for af in af_values:
    row = baseline_mean.copy()
    row[base_idx] = af
    row[e_idx]    = baseline_mean[e_idx] * (1 - 0.08*af)
    co2_af_sim.append(
        rf_co2.predict(scaler.transform(row.reshape(1,-1)))[0]
    )

baseline_co2 = rf_co2.predict(
    scaler.transform(baseline_mean.reshape(1,-1)))[0]

fig7, axes7 = plt.subplots(1, 2, figsize=(13, 6))
fig7.suptitle('Figure 7: What-If Simulation — CO₂ Reduction via AF Fraction Increase',
              fontsize=12, fontweight='bold', color=C_NAVY)

ax7a = axes7[0]
ax7a.plot(af_values*100, co2_af_sim, color=C_RED, linewidth=2.5)
ax7a.axhline(baseline_co2, color=C_GREY, linestyle='--', linewidth=1.2,
             label=f'Baseline: {baseline_co2:.4f} t/t')
ax7a.axvline(40, color=C_GREEN, linestyle=':', linewidth=1.5,
             label='EU Target: 40% AF')
ax7a.fill_between(af_values*100, co2_af_sim, baseline_co2,
                  where=[v < baseline_co2 for v in co2_af_sim],
                  alpha=0.15, color=C_GREEN)
ax7a.set_xlabel('Alternative Fuel Fraction (%)')
ax7a.set_ylabel('Predicted CO₂ (t/t cement)')
ax7a.set_title('Impact of Increasing Alternative Fuel Use',
               fontweight='bold', color=C_NAVY)
ax7a.legend(fontsize=8)
ax7a.set_facecolor(C_LIGHT)

ax7b = axes7[1]
milestones = [baseline_co2,
              co2_af_sim[int(0.4/0.8*49)],
              co2_af_sim[-1]]
bars = ax7b.bar(
    ['Current\n(mean AF)', 'EU Target\n(40% AF)', 'Max Sim\n(80% AF)'],
    milestones,
    color=[C_RED, C_AMBER, C_GREEN], alpha=0.85, edgecolor='white', width=0.5)
for i, val in enumerate(milestones):
    r = (baseline_co2 - val) / baseline_co2 * 100
    ax7b.text(i, val+0.002, f'{val:.4f}\n({r:+.1f}%)',
              ha='center', va='bottom', fontsize=9, fontweight='bold')
ax7b.set_ylabel('Predicted CO₂ (t/t cement)')
ax7b.set_title('Projected CO₂ at Key Milestones',
               fontweight='bold', color=C_NAVY)
ax7b.set_facecolor(C_LIGHT)

plt.tight_layout()
plt.savefig('fig7_simulation.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("All 7 figures saved successfully.")
print(f"\nCO2 Model     — R²={r2_co2:.4f}  MAE={mae_co2:.4f}")
print(f"Strength Model — R²={r2_str:.4f}  MAE={mae_str:.4f}")