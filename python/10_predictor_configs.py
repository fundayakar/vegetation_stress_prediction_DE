import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('germany_model_ready.csv')

le_lc  = LabelEncoder()
le_adm = LabelEncoder()
df['lc_enc']  = le_lc.fit_transform(df['lc_name'])
df['adm_enc'] = le_adm.fit_transform(df['adm1'])

static_cols = ['month', 'lc_enc', 'adm_enc']

states = df['adm1'].unique()
np.random.seed(42)
np.random.shuffle(states)
state_folds = np.array_split(states, 4)

def spatial_cv(X, y, df, state_folds):
    scale_pw = (y==0).sum() / (y==1).sum()
    aucs, aps, f1s = [], [], []
    for fold_states in state_folds:
        ti  = df['adm1'].isin(fold_states).values
        tri = ~ti
        m = xgb.XGBClassifier(n_estimators=200, scale_pos_weight=scale_pw,
                               random_state=42, eval_metric='logloss', verbosity=0)
        m.fit(X[tri], y[tri])
        pr = m.predict_proba(X[ti])[:,1]
        aucs.append(roc_auc_score(y[ti], pr))
        aps.append(average_precision_score(y[ti], pr))
        f1s.append(f1_score(y[ti], (pr>0.5).astype(int), zero_division=0))
    return np.mean(aucs), np.mean(aps), np.mean(f1s)

y = df['stress_1sd'].values

configs = {
    'NDVI-only\npersistence':
        ['ndvi_anomaly_lag1'],
    'Hydroclimate-only\n(no NDVI)':
        ['sm_anomaly_lag1','sm_anomaly_lag2','sm_anomaly_lag3',
         'temp_anomaly_lag1','temp_anomaly_lag2','temp_anomaly_lag3',
         'precip_anomaly_lag1','precip_anomaly_lag2','precip_anomaly_lag3'],
    'Full model\n(NDVI + hydroclimate)':
        ['sm_anomaly_lag1','sm_anomaly_lag2','sm_anomaly_lag3',
         'temp_anomaly_lag1','temp_anomaly_lag2','temp_anomaly_lag3',
         'precip_anomaly_lag1','precip_anomaly_lag2','precip_anomaly_lag3',
         'ndvi_anomaly_lag1'],
}

print(f"{'Configuration':<35} {'ROC-AUC':>8} {'PR-AUC':>8} {'F1':>8}")
print('-'*62)
results = {}
for name, feat_cols in configs.items():
    X = df[feat_cols + static_cols].values
    auc, ap, f1 = spatial_cv(X, y, df, state_folds)
    results[name] = (auc, ap, f1)
    print(f"{name.replace(chr(10),' '):<35} {auc:>8.3f} {ap:>8.3f} {f1:>8.3f}")

# Figure
labels = list(results.keys())
aucs = [results[k][0] for k in labels]
aps  = [results[k][1] for k in labels]
f1s  = [results[k][2] for k in labels]

x = np.arange(len(labels))
width = 0.25
colors = ['#3a7ebf', '#e6a817', '#2d7a4f']

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.bar(x - width, aucs, width, label='ROC-AUC', color=colors[0], edgecolor='none')
ax.bar(x,         aps,  width, label='PR-AUC',  color=colors[1], edgecolor='none')
ax.bar(x + width, f1s,  width, label='F1',      color=colors[2], edgecolor='none')
for i, (a, p, f) in enumerate(zip(aucs, aps, f1s)):
    ax.text(i - width, a + 0.005, f'{a:.3f}', ha='center', va='bottom', fontsize=7.5)
    ax.text(i,         p + 0.005, f'{p:.3f}', ha='center', va='bottom', fontsize=7.5)
    ax.text(i + width, f + 0.005, f'{f:.3f}', ha='center', va='bottom', fontsize=7.5)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0, 1.0)
ax.set_ylabel('Score', fontsize=10)
ax.set_title('Predictor configuration comparison (XGBoost, spatial CV)', fontsize=11, pad=10)
ax.spines[['top','right']].set_visible(False)
ax.legend(fontsize=9, frameon=False)
ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
plt.tight_layout()
plt.savefig('fig_predictor_configs.png', dpi=180, bbox_inches='tight')
plt.close()
print("Saved: fig_predictor_configs.png")
