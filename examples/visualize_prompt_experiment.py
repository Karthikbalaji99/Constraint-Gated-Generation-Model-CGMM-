# examples/create_publication_charts.py

import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set publication-quality style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 13
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

# Load most recent results
result_files = glob.glob("prompt_experiment_results_*.json")
if not result_files:
    print("❌ No results file found.")
    exit(1)

latest_file = sorted(result_files)[-1]
print(f"📊 Loading: {latest_file}")

with open(latest_file, "r") as f:
    results = json.load(f)

# Extract data
data = []
for test in results:
    for model_name, model_results in test["results"].items():
        for prompt_level, result in model_results.items():
            analysis = result["analysis"]
            data.append({
                "Test": test["test_name"],
                "Model": model_name,
                "Prompt Level": prompt_level,
                "Should Block": test["should_block"],
                "Is Safe": analysis["is_safe"],
                "Word Count": analysis["word_count"],
                "Truly Blocked": analysis.get("truly_blocked", False),
                "Is Hedging": analysis.get("is_hedging", False),
                "Has Explanation": analysis.get("has_explanation", False),
            })

df = pd.DataFrame(data)

# Color palette
COLORS = {
    "Baseline": "#E74C3C",  # Red
    "Careful": "#F39C12",   # Orange
    "CGMM-Style": "#3498DB", # Blue
    "CGMM": "#27AE60"       # Green
}

print(f"✅ Loaded {len(df)} data points")
print(f"📊 Creating 6 publication-quality charts...\n")

# ============================================================================
# CHART 1: The Main Finding - Safety Rate Comparison
# ============================================================================

print("📊 Chart 1: Safety Rate by Model and Prompt Level...")

fig, ax = plt.subplots(figsize=(14, 8))

# Calculate safety rates
safety_data = df.groupby(["Model", "Prompt Level"])["Is Safe"].mean() * 100
safety_pivot = safety_data.unstack()

# Order models by best performance
model_order = safety_pivot.max(axis=1).sort_values(ascending=True).index

# Create horizontal bar chart
y_pos = np.arange(len(model_order))
width = 0.25

for i, (prompt_level, color) in enumerate([
    ("Baseline", COLORS["Baseline"]),
    ("Careful", COLORS["Careful"]),
    ("CGMM-Style", COLORS["CGMM-Style"])
]):
    if prompt_level in safety_pivot.columns:
        values = [safety_pivot.loc[model, prompt_level] if model in safety_pivot.index else 0 
                  for model in model_order]
        
        bars = ax.barh(y_pos + i*width, values, width, label=prompt_level, 
                       color=color, alpha=0.85, edgecolor='white', linewidth=1.5)
        
        # Add value labels
        for bar, val in zip(bars, values):
            if val > 5:
                ax.text(val + 2, bar.get_y() + bar.get_height()/2, 
                       f'{val:.0f}%', ha='left', va='center',
                       fontsize=11, fontweight='bold', color='black')

ax.set_yticks(y_pos + width)
ax.set_yticklabels(model_order, fontsize=13)
ax.set_xlabel('Safety Rate (%)', fontweight='bold', fontsize=15)
ax.set_title('Safety Rate by Model and Prompt Strictness\n(Higher = Better)', 
             fontweight='bold', fontsize=17, pad=20)
ax.legend(title='Prompt Level', loc='lower right', frameon=True, 
         fancybox=True, shadow=True, fontsize=13, title_fontsize=14)
ax.axvline(x=100, color='#27AE60', linestyle='--', alpha=0.4, linewidth=2, label='Perfect')
ax.set_xlim(0, 108)
ax.grid(axis='x', alpha=0.3, linewidth=1)

plt.tight_layout()
plt.savefig('chart1_safety_rate_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
print("   ✅ Saved: chart1_safety_rate_comparison.png\n")
plt.close()

# ============================================================================
# CHART 2: The Gap - Best Prompted vs CGMM
# ============================================================================

print("📊 Chart 2: CGMM vs Best Prompted Model...")

fig, ax = plt.subplots(figsize=(12, 8))

# Get best CGMM-Style performance for each model
best_prompted = []
for model in df["Model"].unique():
    if model != "CGMM":
        cgmm_style_safety = df[(df["Model"] == model) & 
                               (df["Prompt Level"] == "CGMM-Style")]["Is Safe"].mean() * 100
        if not pd.isna(cgmm_style_safety):
            best_prompted.append({"Model": model, "Safety": cgmm_style_safety})

best_prompted_df = pd.DataFrame(best_prompted).sort_values("Safety", ascending=True)

# Add CGMM
cgmm_safety = df[df["Model"] == "CGMM"]["Is Safe"].mean() * 100
best_prompted_df = pd.concat([
    best_prompted_df,
    pd.DataFrame([{"Model": "CGMM (Baseline)", "Safety": cgmm_safety}])
], ignore_index=True)

# Colors: Red for <90%, Green for >=90%
colors = [COLORS["CGMM"] if safety >= 95 else COLORS["CGMM-Style"] if safety >= 80 else COLORS["Careful"] 
          for safety in best_prompted_df["Safety"]]

bars = ax.barh(best_prompted_df["Model"], best_prompted_df["Safety"], 
               color=colors, alpha=0.85, edgecolor='white', linewidth=2)

# Add value labels
for bar, val in zip(bars, best_prompted_df["Safety"]):
    ax.text(val + 1.5, bar.get_y() + bar.get_height()/2,
           f'{val:.1f}%', ha='left', va='center',
           fontsize=13, fontweight='bold', color='black')

# Add gap annotation
cgmm_val = best_prompted_df[best_prompted_df["Model"] == "CGMM (Baseline)"]["Safety"].values[0]
best_other = best_prompted_df[best_prompted_df["Model"] != "CGMM (Baseline)"]["Safety"].max()
gap = cgmm_val - best_other

ax.annotate(f'Gap: {gap:.1f}%', 
           xy=(best_other, len(best_prompted_df)-2), 
           xytext=(best_other + 5, len(best_prompted_df)-3),
           fontsize=14, fontweight='bold', color='#E74C3C',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
           arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2))

ax.set_xlabel('Safety Rate (%)', fontweight='bold', fontsize=15)
ax.set_title('Best Prompted Performance vs CGMM\n(All non-CGMM models use strictest "CGMM-Style" prompting)', 
             fontweight='bold', fontsize=17, pad=20)
ax.axvline(x=100, color='#27AE60', linestyle='--', alpha=0.4, linewidth=2, label='Perfect')
ax.axvline(x=cgmm_val, color='#27AE60', linestyle=':', alpha=0.6, linewidth=2, label='CGMM')
ax.set_xlim(0, 108)
ax.grid(axis='x', alpha=0.3, linewidth=1)

plt.tight_layout()
plt.savefig('chart2_cgmm_vs_best_prompted.png', dpi=300, bbox_inches='tight', facecolor='white')
print("   ✅ Saved: chart2_cgmm_vs_best_prompted.png\n")
plt.close()

# ============================================================================
# CHART 3: True Blocking Rate (When Should Block)
# ============================================================================

print("📊 Chart 3: True Blocking Rate...")

fig, ax = plt.subplots(figsize=(14, 8))

# Filter for queries that should block
block_data = df[df["Should Block"] == True].groupby(["Model", "Prompt Level"])["Truly Blocked"].mean() * 100
block_pivot = block_data.unstack()

# Sort by CGMM-Style performance
if "CGMM-Style" in block_pivot.columns:
    model_order_block = block_pivot["CGMM-Style"].fillna(0).sort_values(ascending=True).index
else:
    model_order_block = block_pivot.max(axis=1).sort_values(ascending=True).index

y_pos = np.arange(len(model_order_block))

for i, (prompt_level, color) in enumerate([
    ("Baseline", COLORS["Baseline"]),
    ("Careful", COLORS["Careful"]),
    ("CGMM-Style", COLORS["CGMM-Style"])
]):
    if prompt_level in block_pivot.columns:
        values = [block_pivot.loc[model, prompt_level] if model in block_pivot.index else 0 
                  for model in model_order_block]
        
        bars = ax.barh(y_pos + i*width, values, width, label=prompt_level,
                       color=color, alpha=0.85, edgecolor='white', linewidth=1.5)
        
        for bar, val in zip(bars, values):
            if val > 5:
                ax.text(val + 2, bar.get_y() + bar.get_height()/2,
                       f'{val:.0f}%', ha='left', va='center',
                       fontsize=11, fontweight='bold', color='black')

ax.set_yticks(y_pos + width)
ax.set_yticklabels(model_order_block, fontsize=13)
ax.set_xlabel('True Blocking Rate (%)', fontweight='bold', fontsize=15)
ax.set_title('True Blocking Rate When Information is Insufficient\n(Clean blocks without explanation - Higher = Better)', 
             fontweight='bold', fontsize=17, pad=20)
ax.legend(title='Prompt Level', loc='lower right', frameon=True,
         fancybox=True, shadow=True, fontsize=13, title_fontsize=14)
ax.axvline(x=100, color='#27AE60', linestyle='--', alpha=0.4, linewidth=2)
ax.set_xlim(0, 108)
ax.grid(axis='x', alpha=0.3, linewidth=1)

plt.tight_layout()
plt.savefig('chart3_true_blocking_rate.png', dpi=300, bbox_inches='tight', facecolor='white')
print("   ✅ Saved: chart3_true_blocking_rate.png\n")
plt.close()

# ============================================================================
# CHART 4: Hedging Rate (Says "Blocked" but Explains)
# ============================================================================

print("📊 Chart 4: Hedging Rate...")

fig, ax = plt.subplots(figsize=(14, 8))

hedge_data = df[df["Should Block"] == True].groupby(["Model", "Prompt Level"])["Is Hedging"].mean() * 100
hedge_pivot = hedge_data.unstack()

y_pos = np.arange(len(model_order_block))

for i, (prompt_level, color) in enumerate([
    ("Baseline", COLORS["Baseline"]),
    ("Careful", COLORS["Careful"]),
    ("CGMM-Style", COLORS["CGMM-Style"])
]):
    if prompt_level in hedge_pivot.columns:
        values = [hedge_pivot.loc[model, prompt_level] if model in hedge_pivot.index else 0 
                  for model in model_order_block]
        
        bars = ax.barh(y_pos + i*width, values, width, label=prompt_level,
                       color=color, alpha=0.85, edgecolor='white', linewidth=1.5)
        
        for bar, val in zip(bars, values):
            if val > 5:
                ax.text(val + 2, bar.get_y() + bar.get_height()/2,
                       f'{val:.0f}%', ha='left', va='center',
                       fontsize=11, fontweight='bold', color='black')

ax.set_yticks(y_pos + width)
ax.set_yticklabels(model_order_block, fontsize=13)
ax.set_xlabel('Hedging Rate (%)', fontweight='bold', fontsize=15)
ax.set_title('Hedging Rate: Says "Blocked" but Provides Explanations\n(Lower = Better)', 
             fontweight='bold', fontsize=17, pad=20)
ax.legend(title='Prompt Level', loc='upper right', frameon=True,
         fancybox=True, shadow=True, fontsize=13, title_fontsize=14)
ax.axvline(x=0, color='#27AE60', linestyle='--', alpha=0.4, linewidth=2, label='Ideal')
ax.set_xlim(0, 108)
ax.grid(axis='x', alpha=0.3, linewidth=1)

plt.tight_layout()
plt.savefig('chart4_hedging_rate.png', dpi=300, bbox_inches='tight', facecolor='white')
print("   ✅ Saved: chart4_hedging_rate.png\n")
plt.close()

# ============================================================================
# CHART 5: Response Length When Should Block
# ============================================================================

print("📊 Chart 5: Response Conciseness...")

fig, ax = plt.subplots(figsize=(14, 8))

word_data = df[df["Should Block"] == True].groupby(["Model", "Prompt Level"])["Word Count"].mean()
word_pivot = word_data.unstack()

y_pos = np.arange(len(model_order_block))

for i, (prompt_level, color) in enumerate([
    ("Baseline", COLORS["Baseline"]),
    ("Careful", COLORS["Careful"]),
    ("CGMM-Style", COLORS["CGMM-Style"])
]):
    if prompt_level in word_pivot.columns:
        values = [word_pivot.loc[model, prompt_level] if model in word_pivot.index else 0 
                  for model in model_order_block]
        
        bars = ax.barh(y_pos + i*width, values, width, label=prompt_level,
                       color=color, alpha=0.85, edgecolor='white', linewidth=1.5)
        
        for bar, val in zip(bars, values):
            if val > 20:
                ax.text(val + 10, bar.get_y() + bar.get_height()/2,
                       f'{val:.0f}', ha='left', va='center',
                       fontsize=11, fontweight='bold', color='black')

ax.set_yticks(y_pos + width)
ax.set_yticklabels(model_order_block, fontsize=13)
ax.set_xlabel('Average Word Count', fontweight='bold', fontsize=15)
ax.set_title('Response Length When Information is Insufficient\n(Lower = More Concise = Better)', 
             fontweight='bold', fontsize=17, pad=20)
ax.legend(title='Prompt Level', loc='upper right', frameon=True,
         fancybox=True, shadow=True, fontsize=13, title_fontsize=14)
ax.axvline(x=100, color='#F39C12', linestyle='--', alpha=0.4, linewidth=2, label='Target (100 words)')
ax.grid(axis='x', alpha=0.3, linewidth=1)

plt.tight_layout()
plt.savefig('chart5_response_length.png', dpi=300, bbox_inches='tight', facecolor='white')
print("   ✅ Saved: chart5_response_length.png\n")
plt.close()

# ============================================================================
# CHART 6: Prompt Effectiveness (Improvement from Baseline)
# ============================================================================

print("📊 Chart 6: Prompt Effectiveness...")

fig, ax = plt.subplots(figsize=(12, 8))

# Calculate improvement
improvement_data = []
for model in df["Model"].unique():
    baseline = df[(df["Model"] == model) & (df["Prompt Level"] == "Baseline")]["Is Safe"].mean() * 100
    cgmm_style = df[(df["Model"] == model) & (df["Prompt Level"] == "CGMM-Style")]["Is Safe"].mean() * 100
    
    if not pd.isna(baseline) and not pd.isna(cgmm_style):
        improvement_data.append({
            "Model": model,
            "Baseline": baseline,
            "CGMM-Style": cgmm_style,
            "Improvement": cgmm_style - baseline
        })

improvement_df = pd.DataFrame(improvement_data).sort_values("Improvement", ascending=True)

# Color by improvement magnitude
colors_imp = ['#27AE60' if x > 50 else '#3498DB' if x > 25 else '#F39C12' if x > 0 else '#E74C3C' 
              for x in improvement_df["Improvement"]]

bars = ax.barh(improvement_df["Model"], improvement_df["Improvement"], 
               color=colors_imp, alpha=0.85, edgecolor='white', linewidth=2)

# Add value labels
for bar, val in zip(bars, improvement_df["Improvement"]):
    if abs(val) > 2:
        ax.text(val + (2 if val > 0 else -2), bar.get_y() + bar.get_height()/2,
               f'{val:+.1f}%', ha='left' if val > 0 else 'right', va='center',
               fontsize=13, fontweight='bold', color='black')

ax.set_xlabel('Safety Improvement (%)', fontweight='bold', fontsize=15)
ax.set_title('Prompt Effectiveness: Safety Improvement from Baseline to CGMM-Style\n(Positive = Improvement)', 
             fontweight='bold', fontsize=17, pad=20)
ax.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=2)
ax.grid(axis='x', alpha=0.3, linewidth=1)

# Add annotation
avg_improvement = improvement_df["Improvement"].mean()
ax.text(0.02, 0.98, f'Average Improvement: {avg_improvement:+.1f}%',
       transform=ax.transAxes, ha='left', va='top',
       fontsize=14, fontweight='bold',
       bbox=dict(boxstyle='round,pad=0.8', facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.savefig('chart6_prompt_effectiveness.png', dpi=300, bbox_inches='tight', facecolor='white')
print("   ✅ Saved: chart6_prompt_effectiveness.png\n")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print("="*80)
print("✅ ALL 6 CHARTS CREATED SUCCESSFULLY!")
print("="*80)
print("\n📊 Files created:")
print("   1. chart1_safety_rate_comparison.png")
print("   2. chart2_cgmm_vs_best_prompted.png")
print("   3. chart3_true_blocking_rate.png")
print("   4. chart4_hedging_rate.png")
print("   5. chart5_response_length.png")
print("   6. chart6_prompt_effectiveness.png")
print("\n🎯 All charts are:")
print("   ✓ High resolution (300 DPI)")
print("   ✓ Publication-quality")
print("   ✓ Clean and readable")
print("   ✓ Consistent styling")
print("   ✓ Ready for GitHub/papers")
print()