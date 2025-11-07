import pandas as pd
import os

premno = pd.read_csv('23940898.csv', header=None)
premno = premno[[0]]
premno.drop_duplicates(inplace=True, ignore_index=True)

out_path = os.path.join(os.path.dirname(__file__), 'premno_unique.txt')
with open(out_path, 'w', encoding='utf-8') as f:
    for code in premno[0]:
        f.write(f"{code}\n")