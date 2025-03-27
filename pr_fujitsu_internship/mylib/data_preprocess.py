#All libraries here
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
import os

path = Path(__file__).resolve().parent.parent 
data_path = path.joinpath('data')
result_path = path.joinpath('result')

def parse_xml():
    if os.path.isfile('df'):
        # print('df already exsits')
        return pd.read_pickle('df')
    else:
        tree = ET.parse(data_path/'Posts.xml')
        root = tree.getroot() #specify the most upper attribute in the xml file(=> <post>)
        data = []
        for row in root.findall('row'):
            post_type = row.attrib.get('PostTypeId')
            if post_type == '1':
                id = row.attrib.get('Id')
                body = row.attrib.get('Body', '')
                title = row.attrib.get('Title', '')
                tag = row.attrib.get('Tags', '').replace("<", "").replace(">", ",").strip(",")

                if tag:
                    data.append((id, title, body, tag))
        df = pd.DataFrame(data, columns=['id', 'title', 'body', 'tags'])
        df.to_pickle('df')
        return df

df = parse_xml()

print(df.tail())


