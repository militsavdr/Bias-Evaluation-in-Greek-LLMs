import pandas as pd
from deep_translator import GoogleTranslator


file_path = r"C:\Users\milit\OneDrive\Υπολογιστής\diploma\honest_translate\original_translated\es_template.tsv"
#do the same for the el_template
output_file = r"C:\Users\milit\OneDrive\Υπολογιστής\diploma\honest_translate\original_translated\el3_honest_dataset.tsv"

df = pd.read_csv(file_path, sep="\t")

translator = GoogleTranslator(source='es', target='el')
columns_to_translate = ['template_masked', 'raw', 'identity']

# list to be converted to data frame
translated_data = []


for index, row in df.iterrows():
    translated_row = {}
    
    #translate each column
    for col in columns_to_translate:
        original_text = row[col]
        
        if pd.notna(original_text):
            translated_text = translator.translate(original_text)
        else:
            translated_text = ""

        translated_row[col] = translated_text

        
        print(f"translate {index + 1}/{len(df)} | {col}: {original_text} ➝ {translated_text}")
    
    #stays the same:
    translated_row['number'] = row['number']
    translated_row['category'] = row['category']
    translated_row['type'] = row['type']
    
    translated_data.append(translated_row)

    if (index + 1) % 10 == 0:  # save 10 rows each time
        pd.DataFrame(translated_data).to_csv(output_file, sep="\t", index=False)
        print(f"saved the first {index + 1} rows")


final_df = pd.DataFrame(translated_data)
final_df.to_csv(output_file, sep="\t", index=False)
print(f"Completed and saves as {output_file}")
