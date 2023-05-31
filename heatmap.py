import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# Load the data
with open('results.json', 'r') as f:
    data = json.load(f)
# Unique values for the parameters
gravities = sorted(set(float(k.split(',')[0]) for k in data.keys()))
cart_masses = sorted(set(float(k.split(',')[1]) for k in data.keys()))
pole_masses = sorted(set(float(k.split(',')[2]) for k in data.keys()))
pole_lengths = sorted(set(float(k.split(',')[3]) for k in data.keys()))
sns.set(font_scale=0.65)
# Set up the subplots
fig, axs = plt.subplots(len(cart_masses), len(pole_masses), figsize=(20, 20))
# For each subplot (i.e., pair of pole mass and cart mass)
for i, cart_mass in enumerate(cart_masses):
    for j, pole_mass in enumerate(pole_masses):
        ax = axs[i, j]
        # Prepare a dataframe for the heatmap
        df = pd.DataFrame(index=pole_lengths, columns=gravities)
        # Separate data structure to hold confidence intervals
        ci_data = []
        # Fill the dataframe with the means and confidence intervals
        for gravity in gravities:
            for pole_length in pole_lengths:
                key = f"{gravity},{cart_mass},{pole_mass},{pole_length}"
                mean = data[key]["mean"]
                std = data[key]["std"]
                # Calculate the confidence interval
                ci = 1.96 * (std / np.sqrt(30))
                # Add the mean and confidence interval to the dataframe
                df.loc[pole_length, gravity] = mean
                # Store the confidence interval in the separate dictionary
                ci_data.append(ci)
        # Create the heatmap
        heatmap = sns.heatmap(df.astype(float), ax=ax, annot=True, fmt=".1f", cmap='RdYlGn')
        heatmap.set_xlabel("Gravity")
        heatmap.set_ylabel("Pole Length")
        text_i = 0
        # for text in ax.texts:
        #     text.set_text(text.get_text() + f"±{ci_data[text_i]:.2f}")
        #     text_i += 1
        ax.set_title(f"Cart Mass: {cart_mass}, Pole Mass {pole_mass}")

plt.tight_layout()
plt.savefig("heatmap.png", dpi=300)
