import pickle
import xgboost as xgb
import matplotlib.pyplot as plt

# ── Load saved model ───────────────────────
model = pickle.load(open(r'C:\Users\PC\OneDrive\Desktop\work\home-credit-default-risk\model.pkl', 'rb'))

# ── Plot feature importance ────────────────
xgb.plot_importance(model, max_num_features=15)
plt.title('Top 15 Feature Importances')
plt.tight_layout()
plt.savefig(r'C:\Users\PC\OneDrive\Desktop\work\home-credit-default-risk\feature_importance.png')
plt.show()
print("Saved feature_importance.png")