import pandas as pd
import re
from bs4 import BeautifulSoup


data = pd.read_csv('data/index.csv')

def url_features(url):
    features = {
        'url_length' : len(url),
        'special_char_num' : sum(1 for char in url if char in ['@', '#', '.', '/', '~', ',', '$']),
        'contains_login' : int('login' in url),
        'is_ip' : int(bool(re.match(r'\d+\.\d+\.\d+\.\d+', url))),
    }
    return features

def hmtl_features(path_to_html):
    try:
        with open("data/dataset/" + path_to_html.strip('\''), 'r', encoding='utf-8') as file :
            soup = BeautifulSoup(file, 'html.parser')
            features = {
                'external_link_num' : len(soup.find_all('a', href=lambda x: x and 'http' in x)),
                'iframes_num' : len(soup.find_all('iframe')),
                'script_num' : len(soup.find_all('script')),
                'mailto_num' : len(soup.find_all('mailto:')),
                'external_script_num' : len(soup.find_all('script src')),
                'external_img_num' : len(soup.find_all('img src')),
                'external_style_num' : len(soup.find_all('link href')),
                'suspicious_keyword' : int(any(word in soup.text for word in ['verify', 'update', 'secure']))
            }
    except:
        features = {}
        print(path_to_html + " not found :(")
    return features
    
features_list = []
counter = 0
for index, row in data.iterrows():
    counter += 1
    print("parsing row " + str(row['rec_id']))
    url_feature = url_features(row['url'])
    html_feature = hmtl_features(row['website'])
    combined_features = {**url_feature, **html_feature, 'label': row['result']}
    features_list.append(combined_features)
    if counter == 100 :
        break
features_df = pd.DataFrame(features_list)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(features_df.to_string())