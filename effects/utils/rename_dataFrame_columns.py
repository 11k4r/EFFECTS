import re

def rename_dataFrame_columns(df):
    df.columns = [re.sub('_| ', '', column) for column in df.columns]
    return df