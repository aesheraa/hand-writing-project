




# MACHINE LEARNING PROJECT

# ## Foto�raflardaki El Yaz�s� Rakamlar� yapay zeka ile Otomatik Tan�ma ve Anlamland�rma


#Birden fazla makine ��renmesi modelini bir arada kullanarak foto�raflardaki objeleri tan�yan ve anlamland�ran bir yaz�l�m ger�ekle�tirece�im.
#�nce el yz�s�yla yaz�lan rakamlar�n foto�raflar�n� sisteme y�kleyip bir rakam i�in sistemimi e�itece�im daha sonra sisteme el yaz�s�yla yaz�lan yeni bir rakam� tan�mas�n� isteyece�im. 
#Bu projede iki farkl� machine learning modeli kullanaca��m PCA ve logistic regression. 

#Elimde 70 bin tane elyaz�s�yla yaz�lm�� rakam foto�raf� var. 70 bin veriden olu�an verisetinde 60 bin veri modeli e�itmek i�in, 10 bin veri test etmek i�in kullanaca��m. 
#verilerin Her biri 28 e 28 pikselden olu�an resimler. (yani iki matrisli, toplam 784 feature ) 
#1. a�amada PCA i�lemi yapt�m 2. a�amada logistic regression modelimizi PCA  i�leminden ge�irilmi� verisetimize uygulayaca��m.

# Projemizde kullanaca��m�z MNIST elyaz�s� rakamlar� veritaban�nda(sklearn i�inde gelmektedir) 784 feature s�tunu mevcut (784 dimensions), 
# ve training set olarak 60,000 �rnek veri ve a 10,000 �rneklik test seti bulunmaktad�r.



import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml  # mnist datasetini y�klemek i�in gerekli...
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# Bu i�lem 1-2 dk s�rebilir..
mnist = fetch_openml('mnist_784')


mnist.data.shape


# Mnist veriseti i�indeki rakam foto�raflar�n� g�rmek i�in bir fonksiyon tan�mlayal�m:


# Parametre olarak dataframe ve ilgili veri foto�raf�n�n index numaras�n� als�n..
def showimage(dframe, index):    
    some_digit = dframe.to_numpy()[index]
    some_digit_image = some_digit.reshape(28,28)

    plt.imshow(some_digit_image,cmap="binary")
    plt.axis("off")
    plt.show()




# �rnek kullan�m:
showimage(mnist.data, 0)


# 70,000 image dosyas�, her bir image i�in 784 boyut(784 feature) mevcut.

# Split Data -> Training Set ve Test Set




# test ve train oran� 1/7 ve 6/7
train_img, test_img, train_lbl, test_lbl = train_test_split( mnist.data, mnist.target, test_size=1/7.0, random_state=0)


type(train_img)



# Rakam tahminlerimizi check etmek i�in train_img dataframeini kopyal�yoruz, ��nk� az sonra de�i�ecek..
test_img_copy = test_img.copy()




showimage(test_img_copy, 2)


# ### Verilerimizi Scale etmemiz gerekiyor:
# ��nk� PCA scale edilmemi� verilerde hatal� sonu�lar verebiliyor bu nedenle mutlaka scaling i�leminden ge�iriyoruz. 
# Bu ama�la da StandardScaler kullan�yoruz...



scaler = StandardScaler()

# Scaler'� sadece training set �zerinde fit yapmam�z yeterli..
scaler.fit(train_img)

# Ama transform i�lemini hem training sete hem de test sete yapmam�z gerekiyor..
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)


#  PCA i�lemini uyguluyoruz..

# Variance'�n 95% oran�nda korunmas�n� istedi�imizi belirtiyoruz..



# Make an instance of the Model
pca = PCA(.95)



# PCA'i sadece training sete yapmam�z yeterli: (1 dk s�rebilir)
pca.fit(train_img)



# Bakal�m 784 boyutu ka� boyuta d���rebilmi� (%95 variance'� koruyarak tabiiki)
print(pca.n_components_)



# �imdi transform i�lemiyle hem train hem de test veri setimizin boyutlar�n� 784'ten 327'e d���relim:
train_img = pca.transform(train_img)
test_img = pca.transform(test_img)


# 2. A�ama
# �imdi 2. Makine ��renmesi modelimiz olan Logistic Regression modelimizi PCA i�leminden ge�irilmi� veris etimiz �zerinde uygulayaca��z.



# default solver �ok yava� �al��t��� i�in daha h�zl� olan 'lbfgs' solver� se�erek logisticregression nesnemizi olu�turuyoruz.
logisticRegr = LogisticRegression(solver = 'lbfgs', max_iter=10000)


# LogisticRegression Modelimizi train datam�z� kullanarak e�itiyoruz:


# (Birka� dk s�rebilir)
logisticRegr.fit(train_img, train_lbl)


#Modelimiz e�itildi �imdi el yaz�s� rakamlar� makine ��renmesi ile tan�ma i�lemini ger�ekletirelim:


logisticRegr.predict(test_img[0].reshape(1,-1))




showimage(test_img_copy, 0)



logisticRegr.predict(test_img[1].reshape(1,-1))




showimage(test_img_copy, 1)



showimage(test_img_copy, 9900)



logisticRegr.predict(test_img[9900].reshape(1,-1))




showimage(test_img_copy, 9999)



logisticRegr.predict(test_img[9999].reshape(1,-1))



# ### Modelimizin do�ruluk oran� (accuracy) �l�mek 


# Modelimizin do�ruluk oran� (accuracy) �l�mek i�in score metodunu kullanaca��z:

logisticRegr.score(test_img, test_lbl)


# ### Sonu� ve De�erlendirme

# Bu projede PCA kullanarak logistic regression taraf�ndan yapay zekan�n e�itilme s�resini �nemli �l��de k�saltt�k. 
# Ben %95 variance korumay� hedefledim. Siz % 95  variance'� daha d���k seviyelere �ekerek s�renin ne �l��de k�salt�d���n� kendiniz deneyerek bulabilirsiniz. 
# 10 tane digit i�in yapay zekan�n e�itim s�resini �ok b�y�k �l��de k�saltan PCA algoritmas� 
# y�zlerce hatta binlerce de�i�ik nesne tipi i�in yapay zekan�n e�itim s�resini saatler mertebesinde k�saltacak ve bu da sizin programlar�n�z�n �ok daha h�zl� �al��mas�n� sa�layacakt�r.

