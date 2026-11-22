from pathlib import Path
import pandas as pd
from graph import build_graph

ROOT = Path(__file__).resolve().parent
for name in ['demo_classification.csv','demo_regression.csv']:
    df = pd.read_csv(ROOT / name)
    state = {
        'df': df.copy(), 'source_df': df.copy(),
        'goal': 'Analyze the dataset, find important patterns, and build a suitable prediction model.',
        'target_override': None, 'llm_provider': 'offline', 'llm_model': None,
        'model_search_mode': 'fast', 'profile': {}, 'preprocessing': {}, 'plan': [],
        'task': '', 'target': None, 'decision': {}, 'eda': {}, 'model': {},
        'validation': {}, 'insights': '', 'logs': []
    }
    out = build_graph().invoke(state)
    p = out.get('processed_df')
    print('\nDATASET:', name)
    print('TASK:', out.get('task'))
    print('TARGET:', out.get('target'))
    print('PROCESSED MISSING CELLS:', int(p.isna().sum().sum()) if p is not None else 'missing')
    print('MODEL STATUS:', out.get('model', {}).get('status'))
    print('MODELS EVALUATED:', out.get('model', {}).get('models_evaluated'))
    print('SAMPLED:', out.get('model', {}).get('sampled'))
    print('INSIGHTS:', bool(out.get('insights')))
    print('LAST LOG:', out.get('logs', [])[-1] if out.get('logs') else 'none')
print('\nValidation complete.')
