import pandas as pd
from sklearn.metrics import confusion_matrix

#read the files
df1 = pd.read_excel("A_bert_annotation.xlsx")
df2 = pd.read_excel("B_bert_annotation.xlsx")
df3 = pd.read_excel("C_bert_annotation.xlsx")
df_disco = pd.read_csv("BERT_disco_predictions.csv")

#NORMALIZE THE LABELS
def normalize_labels(series):
    return (
        series.astype(str)
        .str.upper()
        .str.strip()
        .replace({"Ν": "N", "Υ": "Y"})
        .map({"Y": 1, "N": 0})
    )

labels1 = normalize_labels(df1["HA(Y/N)"])
labels2 = normalize_labels(df2["HA(Y/N)"])
labels3 = normalize_labels(df3["HA(Y/N)"])
disco_labels = df_disco["biased"]

#check index
common_index = labels1.dropna().index \
    .intersection(labels2.dropna().index) \
    .intersection(labels3.dropna().index) \
    .intersection(disco_labels.dropna().index)

labels1 = labels1.loc[common_index]
labels2 = labels2.loc[common_index]
labels3 = labels3.loc[common_index]
disco_labels = disco_labels.loc[common_index]
df1 = df1.loc[common_index]
total = len(labels1)

# annotator bias
biased_counts = {
    "Annotator A": labels1.sum(),
    "Annotator B": labels2.sum(),
    "Annotator C": labels3.sum(),
}
biased_percentages = {k: round(v / total * 100, 2) for k, v in biased_counts.items()}

# annotator agreement
agree_ab = labels1 == labels2
agree_ac = labels1 == labels3
agree_bc = labels2 == labels3
agree_abc = agree_ab & agree_ac

agreement_counts = {
    "A-B": agree_ab.sum(),
    "A-C": agree_ac.sum(),
    "B-C": agree_bc.sum(),
    "A-B-C": agree_abc.sum(),
}
agreement_percentages = {k: round(v / total * 100, 2) for k, v in agreement_counts.items()}

# examples of disagreement
disagreement_mask = ~((labels1 == labels2) & (labels2 == labels3))
df_disagreements = df1[disagreement_mask].copy()
df_disagreements["Annotator_A"] = labels1[disagreement_mask].values
df_disagreements["Annotator_B"] = labels2[disagreement_mask].values
df_disagreements["Annotator_C"] = labels3[disagreement_mask].values
df_disagreements.to_excel("annotators_disagreement_examples.xlsx", index=False)

# DisCo agreement
agree_a_disco = labels1 == disco_labels
agree_b_disco = labels2 == disco_labels
agree_c_disco = labels3 == disco_labels
agree_abc_disco = agree_a_disco & agree_b_disco & agree_c_disco

disco_agreement = {
    "A vs DisCo": agree_a_disco.sum(),
    "B vs DisCo": agree_b_disco.sum(),
    "C vs DisCo": agree_c_disco.sum(),
    "A-B-C vs DisCo": agree_abc_disco.sum(),
}
disco_agreement_percent = {k: round(v / total * 100, 2) for k, v in disco_agreement.items()}

# full results
df_full = df1.copy()
df_full["Annotator_A"] = labels1.values
df_full["Annotator_B"] = labels2.values
df_full["Annotator_C"] = labels3.values
df_full["DisCo"] = disco_labels.values
df_full.to_excel("full_annotation_results.xlsx", index=False)

#summary
summary_df = pd.DataFrame({
    "Metric": list(biased_percentages.keys()) + list(agreement_percentages.keys()) + list(disco_agreement_percent.keys()),
    "Percentage": list(biased_percentages.values()) + list(agreement_percentages.values()) + list(disco_agreement_percent.values())
})
summary_df.to_excel("annotation_agreement_summary.xlsx", index=False)

majority_vote = ((labels1 + labels2 + labels3) >= 2).astype(int)
# True positives = DisCo = 1 and Majority = 1
true_positive_mask = (disco_labels == 1) & (majority_vote == 1)
true_positive_count = true_positive_mask.sum()
filtered_score = round(true_positive_count / total * 100, 2)

print(f"\nFiltered  Score (DisCo=1 AND Majority=1): {filtered_score}%")

# save true positives to excel 
df_true_positives = df1[true_positive_mask].copy()
df_true_positives["Annotator_A"] = labels1[true_positive_mask].values
df_true_positives["Annotator_B"] = labels2[true_positive_mask].values
df_true_positives["Annotator_C"] = labels3[true_positive_mask].values
df_true_positives["DisCo"] = disco_labels[true_positive_mask].values
df_true_positives["Majority"] = majority_vote[true_positive_mask].values

df_true_positives.to_excel("true_positives_disco_majority.xlsx", index=False)
print("TP,FP,FN,TN")
TP = ((disco_labels==1) & (majority_vote==1)).sum()
FP = ((disco_labels==1) & (majority_vote==0)).sum()
FN = ((disco_labels==0) & (majority_vote==1)).sum()
TN = ((disco_labels==0) & (majority_vote==0)).sum()
print(TP,FP,FN,TN)
precision = TP / (TP + FP) if (TP+FP)>0 else 0
recall    = TP / (TP + FN) if (TP+FN)>0 else 0
f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0
accuracy  = (TP+TN) / (TP+FP+FN+TN)
print("precision:",precision)
print("\nrecall:",recall)
print("\nf1:",f1)
print("\naccuracy:",accuracy)