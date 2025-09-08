import pandas as pd

input_file = r"C:\Users\milit\OneDrive\Υπολογιστής\diploma\honest_translate\modify_similarity\honest_similarity_results_EN_ES.tsv"  
output_file =r"C:\Users\milit\OneDrive\Υπολογιστής\diploma\honest_translate\modify_similarity\no_match_EN_ES.tsv"  

df = pd.read_csv(input_file, sep="\t")

df_mismatches = df[df["match"] == 1]

df_mismatches.to_csv(output_file, sep="\t", index=False)

print(len(df_mismatches))
