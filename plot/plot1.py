import numpy as np
import matplotlib.pyplot as plt

data = {
    "GPT-4o": [
        [9.39, 8.98, 9.56, 9.33, 6.99],
        [9.36, 8.94, 9.58, 9.32, 7.23],
        [9.40, 8.99, 9.64, 9.38, 6.91]
    ],
    "GPT-4.1": [
        [9.53, 9.03, 9.56, 9.34, 8.08],
        [9.53, 9.01, 9.59, 9.33, 8.04],
        [9.50, 8.98, 9.62, 9.34, 8.08]
    ],
    "DeepSeek-V3": [
        [9.52, 9.01, 9.20, 9.20, 8.30],
        [9.53, 9.03, 9.13, 9.18, 8.29],
        [9.54, 9.03, 9.17, 9.23, 8.34]
    ],
    "DeepSeek-R1": [
        [9.61, 9.18, 9.02, 9.26, 8.19],
        [9.61, 9.18, 9.11, 9.27, 8.17],
        [9.61, 9.19, 9.11, 9.28, 8.27]
    ]
}

averages = {model: np.mean(np.array(scores), axis=0) for model, scores in data.items()}

dims = ["Answer Relevance", "Context Precision", "Completeness", "Coherency", "Formatting Quality"]
x = np.arange(len(dims))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 6))
for i, (model, scores) in enumerate(averages.items()):
    bars = ax.bar(x + i * width, scores, width, label=model)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

# ax.set_xticks(x + width * (len(averages) - 1) / 2 + 0.25)
# ax.set_xticklabels(dims, rotation=30, ha='right')

ax.set_xticks(x + width * (len(averages) - 1) / 2)  
ax.set_xticklabels(dims, rotation=20, ha='center')



ax.set_ylabel("Average Score Across Three Trials")
ax.set_ylim(7, 10)
ax.set_title("Average LLM-based Evaluation Scores per Dimension for Four LLMs Using DeepSeek-V3 as Evaluator")
ax.legend()

plt.tight_layout()
plt.savefig("avg_scores.pdf", bbox_inches='tight')
plt.show()
