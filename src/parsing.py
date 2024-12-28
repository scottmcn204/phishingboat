import pandas as pd
import re
from bs4 import BeautifulSoup

def url_features(url):
    features = {
        'url_length' : len(str(url)),
        'special_char_num' : sum(1 for char in str(url) if char in ['@', '#', '/', '~', ',', '$']),
        'contains_login' : int('login' in url),
        'numbers_num' : sum(1 for char in url if char in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']),
        'is_http': int("http://" in url),
        'dots_num' :  sum(1 for char in url if char in ['.']),
    }
    return features

def hmtl_features(path_to_html):
    try:
        with open("data/dataset/" + path_to_html.strip('\''), 'r', encoding='utf-8') as file :
            soup = BeautifulSoup(file, 'html.parser')
            html = soup.prettify( formatter="html" )
            features = {
                'external_link_num' : len(soup.find_all('a', href=lambda x: x and 'http' in x)),
                'iframes_num' : len(soup.find_all('iframe')),
                'script_num' : len(soup.find_all('script')),
                'external_script_num' : len(soup.find_all('script', src=True)),
                'external_img_num' : len(soup.find_all('img', src=True)),
                'external_style_num' : len(soup.find_all('link', href=True)),
                'suspicious_keyword' : int(any(word in soup.text for word in ['verify', 'update', 'secure', 'gift', 'free', 'promotion', 'win', 'prize', 'virus'])),
                'overall_length' : len(soup),
            }
    except:
        features = {}
        html = None
        print(path_to_html + " not found :(")
    return features, html

def main(): 
    data = pd.read_csv('data/index.csv')
    features_list = []
    counter = 0
    for index, row in data.iterrows():
        counter += 1
        print("parsing row " + str(row['rec_id']))
        url_feature = url_features(row['url'])
        html_feature, html_raw = hmtl_features(row['website'])
        combined_features = {**url_feature, **html_feature, 'label': row['result'],
        'html_raw': html_raw}
        features_list.append(combined_features)
        # if counter == 100 :
        #     break
    features_df = pd.DataFrame(features_list)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(features_df.to_string())
    return features_df

features_df = main()
features_df.to_csv('features.csv') 


# rare event prediction