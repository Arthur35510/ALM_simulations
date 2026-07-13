import pandas as pd
df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
toto = df.agg({"col1":"mean", "col2":"mean"}).to_dict()
print(toto)
print(type(toto))
