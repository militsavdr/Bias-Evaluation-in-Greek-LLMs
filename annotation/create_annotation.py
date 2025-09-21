import pandas as pd

df = pd.read_csv('socialbert_paper_disco_predictions.csv')

cols_to_keep = [
    'sentence_male',
    'sentence_female',
    'predictions_male',
    'predictions_female',
    
]
df_new = df[cols_to_keep]

df_new['HA(Y/N)'] = ''
df_new['Comment']=''

df_new.to_excel('socialbert_annotation.xlsx', index=False)
