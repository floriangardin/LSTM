import numpy as np
from ipdb import set_trace as pause
from pyquery import PyQuery as pq
import wikipedia as wiki
import sentiment
from sklearn.feature_extraction import FeatureHasher as fh
# Get a page randomly in wikipedia




if __name__ == '__main__':
    # We get the html of a random page on wikipedia
    random_page = 'https://en.wikipedia.org/wiki/special:random'
    page= pq(url=random_page).html()

    # Create features and retrieve the original page
    features = map(ord,page)
    page_copy = map(unichr,features)
    page_copy = ''.join(page_copy)

    # instantiate a rnn or lstm




    pause()