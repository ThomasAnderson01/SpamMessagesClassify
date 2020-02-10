from data_process import data_process
from wordcloud import WordCloud
import matplotlib.pyplot as plt

adata, data_after_stop, labels = data_process()

word_fre = {}
for i in data_after_stop[labels == 0]:
    for j in i:
        if j not in word_fre.keys():
            word_fre[j] = 1
        else:
            word_fre[j] += 1

mask = plt.imread('duihuakuan.jpg')
wc = WordCloud(mask=mask, background_color='white', font_path=r'C:\Windows\Fonts\simhei.ttf')
wc.fit_words(word_fre)
plt.imshow(wc)
plt.show()