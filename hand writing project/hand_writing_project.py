




# MACHINE LEARNING PROJECT

# ## Fotoðraflardaki El Yazýsý Rakamlarý yapay zeka ile Otomatik Tanýma ve Anlamlandýrma


#Birden fazla makine öðrenmesi modelini bir arada kullanarak fotoðraflardaki objeleri tanýyan ve anlamlandýran bir yazýlým gerçekleþtireceðim.
#Önce el yzýsýyla yazýlan rakamlarýn fotoðraflarýný sisteme yükleyip bir rakam için sistemimi eðiteceðim daha sonra sisteme el yazýsýyla yazýlan yeni bir rakamý tanýmasýný isteyeceðim. 
#Bu projede iki farklý machine learning modeli kullanacaðým PCA ve logistic regression. 

#Elimde 70 bin tane elyazýsýyla yazýlmýþ rakam fotoðrafý var. 70 bin veriden oluþan verisetinde 60 bin veri modeli eðitmek için, 10 bin veri test etmek için kullanacaðým. 
#verilerin Her biri 28 e 28 pikselden oluþan resimler. (yani iki matrisli, toplam 784 feature ) 
#1. aþamada PCA iþlemi yaptým 2. aþamada logistic regression modelimizi PCA  iþleminden geçirilmiþ verisetimize uygulayacaðým.

# Projemizde kullanacaðýmýz MNIST elyazýsý rakamlarý veritabanýnda(sklearn içinde gelmektedir) 784 feature sütunu mevcut (784 dimensions), 
# ve training set olarak 60,000 örnek veri ve a 10,000 örneklik test seti bulunmaktadýr.



import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml  # mnist datasetini yüklemek için gerekli...
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# Bu iþlem 1-2 dk sürebilir..
mnist = fetch_openml('mnist_784')


mnist.data.shape


# Mnist veriseti içindeki rakam fotoðraflarýný görmek için bir fonksiyon tanýmlayalým:


# Parametre olarak dataframe ve ilgili veri fotoðrafýnýn index numarasýný alsýn..
def showimage(dframe, index):    
    some_digit = dframe.to_numpy()[index]
    some_digit_image = some_digit.reshape(28,28)

    plt.imshow(some_digit_image,cmap="binary")
    plt.axis("off")
    plt.show()




# Örnek kullaným:
showimage(mnist.data, 0)


# 70,000 image dosyasý, her bir image için 784 boyut(784 feature) mevcut.

# Split Data -> Training Set ve Test Set




# test ve train oraný 1/7 ve 6/7
train_img, test_img, train_lbl, test_lbl = train_test_split( mnist.data, mnist.target, test_size=1/7.0, random_state=0)


type(train_img)



# Rakam tahminlerimizi check etmek için train_img dataframeini kopyalýyoruz, çünkü az sonra deðiþecek..
test_img_copy = test_img.copy()




showimage(test_img_copy, 2)


# ### Verilerimizi Scale etmemiz gerekiyor:
# Çünkü PCA scale edilmemiþ verilerde hatalý sonuçlar verebiliyor bu nedenle mutlaka scaling iþleminden geçiriyoruz. 
# Bu amaçla da StandardScaler kullanýyoruz...



scaler = StandardScaler()

# Scaler'ý sadece training set üzerinde fit yapmamýz yeterli..
scaler.fit(train_img)

# Ama transform iþlemini hem training sete hem de test sete yapmamýz gerekiyor..
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)


#  PCA iþlemini uyguluyoruz..

# Variance'ýn 95% oranýnda korunmasýný istediðimizi belirtiyoruz..



# Make an instance of the Model
pca = PCA(.95)



# PCA'i sadece training sete yapmamýz yeterli: (1 dk sürebilir)
pca.fit(train_img)



# Bakalým 784 boyutu kaç boyuta düþürebilmiþ (%95 variance'ý koruyarak tabiiki)
print(pca.n_components_)



# Þimdi transform iþlemiyle hem train hem de test veri setimizin boyutlarýný 784'ten 327'e düþürelim:
train_img = pca.transform(train_img)
test_img = pca.transform(test_img)


# 2. Aþama
# Þimdi 2. Makine Öðrenmesi modelimiz olan Logistic Regression modelimizi PCA iþleminden geçirilmiþ veris etimiz üzerinde uygulayacaðýz.



# default solver çok yavaþ çalýþtýðý için daha hýzlý olan 'lbfgs' solverý seçerek logisticregression nesnemizi oluþturuyoruz.
logisticRegr = LogisticRegression(solver = 'lbfgs', max_iter=10000)


# LogisticRegression Modelimizi train datamýzý kullanarak eðitiyoruz:


# (Birkaç dk sürebilir)
logisticRegr.fit(train_img, train_lbl)


#Modelimiz eðitildi þimdi el yazýsý rakamlarý makine öðrenmesi ile tanýma iþlemini gerçekletirelim:


logisticRegr.predict(test_img[0].reshape(1,-1))




showimage(test_img_copy, 0)



logisticRegr.predict(test_img[1].reshape(1,-1))




showimage(test_img_copy, 1)



showimage(test_img_copy, 9900)



logisticRegr.predict(test_img[9900].reshape(1,-1))




showimage(test_img_copy, 9999)



logisticRegr.predict(test_img[9999].reshape(1,-1))



# ### Modelimizin doðruluk oraný (accuracy) ölçmek 


# Modelimizin doðruluk oraný (accuracy) ölçmek için score metodunu kullanacaðýz:

logisticRegr.score(test_img, test_lbl)


# ### Sonuç ve Deðerlendirme

# Bu projede PCA kullanarak logistic regression tarafýndan yapay zekanýn eðitilme süresini önemli ölçüde kýsalttýk. 
# Ben %95 variance korumayý hedefledim. Siz % 95  variance'ý daha düþük seviyelere çekerek sürenin ne ölçüde kýsaltýdýðýný kendiniz deneyerek bulabilirsiniz. 
# 10 tane digit için yapay zekanýn eðitim süresini çok büyük ölçüde kýsaltan PCA algoritmasý 
# yüzlerce hatta binlerce deðiþik nesne tipi için yapay zekanýn eðitim süresini saatler mertebesinde kýsaltacak ve bu da sizin programlarýnýzýn çok daha hýzlý çalýþmasýný saðlayacaktýr.

