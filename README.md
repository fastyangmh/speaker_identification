#   speaker_identification
[Google Colab](https://colab.research.google.com/drive/12sKidXvTXbzm6jFK5WVz6vf6JanoMgz2?authuser=1)
##  Abstract

    此專案目的為語者辨識，給予語者MFCC特徵，利用卷積神經網路的特性抓取語者重要特徵做語者辨識，下為模型效能。
    Trainset accuracy: 0.9456, unclassifiable: 0.0381
    Testset accuracy: 0.9084, unclassifiable: 0.0528
    3 second accuracy: 0.7855, unclassifiable: 0.1038
    1 second accuracy: 0.5692, unclassifiable: 0.1786
    imposter accuracy: 0.3173, unclassifiable: 0.3173

    此專案中使用的語料來源於愛丁堡大學製作的CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit，此語料包含110位各種口音的英語人士所錄製，在本專案中從中挑選30位語者進行實驗。

## Model
    此模型使用一層卷積層和一層池化層並加上一層全連接層，最後使用softmax輸出所有語者的機率。

##  Train
    語料經由挑選後，得到的資料量為11861筆，平均每位語者擁有395筆語音，每筆語音時間長度約為3秒，並經由MFCC轉換取得39維特徵。
    
    data.shape=(11861, 39, 180)

##  Predict
    給予特徵，模型便會輸出所有語者的機率，接著從中挑選最大的機率同時要滿足大於預先設定的閾值(0.5)，才會屬於該語者，若不滿足則該特徵並不屬於任何語者，則可以視為仿冒者(imposter)。
    此模型要盡可能地辨識出所有語者，並且能夠分辨出仿冒者。
