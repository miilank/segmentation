Pretpostavka od koje sam krenuo:
- Naglasavanje ivica objekata
- Segmentacija
- Morfolosko ciscenje (erozija, dilacija)

Prateci https://youtu.be/oXlwWbU8l2o?t=3426 dosao sam do pitanja da li ima razlike izmedju
Threshold i Canny (https://stackoverflow.com/questions/29322188/canny-edge-vs-thresholding-for-contour-estimation-in-open-cv).
Prvi korak je bio prebacivanje slike u sivu i naglasavanje ivica (Canny) kao priprema za segmentaciju.

Ivica je mjesto nagle promjene svjetline (izlaz Canny-ja), a kontura je lista tacaka, tj. piksela
koji leze na granici jednog bijelog regiona. Ako krenemo od jedne tacke i idemo po njima, obici cemo
cijeli region i vratiti se gdje smo krenuli. Bijeli region pravi threshold, a findContours vraca listu koordinata tacaka na granici tog regiona. Canny daje linije,
a ne popunjene regione, pa povrsina dobijenih kontura nema smisla.

Canny sam izabrao jer banane i oblaci u sivoj slici imaju slicnu svjetlinu, pa sam ocekivao da
obican prag nece moci da ih razdvoji, a da ce Canny izdvojiti samo jake ivice. Kao sto je objasnjeno:

> "Canny Edge Detection obviously. It does a whole bunch of things to ensure that only strong edges come out of the result. Thresholding just takes a look at intensities and sees whether or not each value is smaller or larger and we get "edge" points respectively"

- Sa sivom slikom: mnogo banana se izgubi, jer pretvaranjem u sivu gubim boju,
  a zuta banana u sivom ima slicnu svjetlinu kao pozadina (nebo, trava), pa nema jaku ivicu.
  Iz ovoga zakljucujem da siva slika nije dobar ulaz, odnosno da mi treba informacija o boji.
- Bez sive slike: naglase se SVE ivice (lisce, trava, stabla), pa se banane
  izgube medju silnim konturama.

Zakljucak: Canny reaguje na promjene svjetline, a ne na boju. Kod ovih slika je
boja jedino sto jasno izdvaja banane, pa segmentaciju
radim po boji (HSV maska), a ne po ivicama.

Kod kojim sam testirao (prema gorenavedenom videu):
```python
import cv2 as cv
import numpy as np

img = cv.imread('data/images/image_1.jpg')
cv.imshow('img', img) # 3 kanala, cv.imread i cv.imshow rade u BGR - konverzija treba samo za matplotlib (RGB)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blank = np.zeros(img.shape, dtype='uint8')

# 1. nacin: Canny
blur = cv.GaussianBlur(gray, (5, 5), 0)
canny = cv.Canny(blur, 125, 175)
cv.imshow('Canny', canny) # 1 kanal, ovdje BGR ili RGB prikaz, nije bitno 

# 2. nacin: Threshold
ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
cv.imshow('Thresh', thresh)

# 1. nacin: proslijedimo canny kao prvi argument
# 2. nacin: proslijedimo thresh kao prvi argument
contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contours found.')

cv.drawContours(blank, contours, -1, (0, 0, 255), 1)
cv.imshow('Contours Drawn', blank)

cv.waitKey(0)
cv.destroyAllWindows()
```

Kao sto je navedeno, boja mi je bitna. S tim u vezi, koristim HSV i maskiranje. U vjezbama 1 pise:

> Sve metode koje smo do sada radili rade nad grayscale slikom. Konverzija u grayscale je prvi korak, i već u njemu gubimo informaciju o boji. Ako nam je boja bitna, potreban nam je prostor boja u kojem je ona izdvojena kao posebna veličina.

Bijeli region pravi threshold (po svjetlini - da li si svjetliji od 125, ako jesi postajes bijel, ako nisi, postajes crn) ili HSV maska (po boji - da li je piksel boje izmedju donje i gornje, ako jeste, postaje bijel, ako nije, postaje crn). U oba slucaja izlaz je crna pozadina sa bijelim mrljama, samo je bolji HSV, jer on pita, da li je zuto, a ne da li je svijetlo.

Maska za HSV je slicno sto i Threshold za sivu sliku(ne bukvalno receno, razlike naravno ima, ali cisto radi sagledavanja sire slike).
BitwiseAnd radim da bi na stvarnoj slici gdje je bijelo ostala normalna boja, a gdje je crno, da bude crno. Tako se izdvajaju banane. 


Samo treba pogoditi zutu boju. Kako? 
Rjesenje sam potrazio ovdje:
Za svoj program sam izabrao Interactive Trackbar, kao na linku: https://medium.com/@sardorabdirayimov/lower-and-upper-mask-range-for-color-detection-opencv-7f64a5df568b

AI je koristen za prilagodjavanje Trackbar-a u radu sa slikama.

```python
# maska ima 1 broj po pikselu (1 kanal), pa plt ne zna kojom bojom da ga prikaze - cmap mu kaze pravilo.
# Bez cmap se koristi podrazumijevana mapa (viridis): najmanji broj na slici - ljubicasta, srednji - zelena, najveci - zuta
# Sa cmap='gray': najmanji - crna, najveci - bijela, izmedju - nijanse sive
plt.imshow(mask, cmap='gray')
```