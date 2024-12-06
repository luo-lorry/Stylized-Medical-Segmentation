import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

labels = ['Dice', 'IoU', 'Specificity', r'$\mathregular{F_\beta^w}$',
          r'$\mathregular{S_\alpha}$', r'$\mathregular{E_\phi^{max}}$', '1-MAE']

data = {
    'UNet': [0.6692, 0.5661, 0.8609, 0.2543, 0.9394, 0.6905, 1-0.1411],
    'UNet(ST)': [0.7063, 0.6014, 0.9133, 0.3275, 0.9592, 0.8242, 1-0.0945]
}

markers = ['D', '^']
colors = ['#FF8080', '#33CC33']

angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))

fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))

for i, (algorithm, values) in enumerate(data.items()):
    values = np.concatenate((values, [values[0]]))
    ax.plot(angles, values, 'o-', color=colors[i], label=algorithm,
            marker=markers[i], markersize=12, linewidth=2, alpha=0.7)
    ax.fill(angles, values, color=colors[i], alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=20)

ax.grid(True, linestyle='--', alpha=0.7)

ax.set_ylim(0.2, 1)

plt.legend(loc='upper left', bbox_to_anchor=(0, 0.95), fontsize=15)

plt.tight_layout()

if not os.path.exists('./Radar chart'):
    os.makedirs('./Radar chart')

plt.savefig('./Radar chart/Radar image of Polyp-UNet dataset.pdf', format='pdf', bbox_inches='tight', dpi=300)