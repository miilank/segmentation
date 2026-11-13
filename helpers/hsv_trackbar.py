# Osnova preuzeta sa:
# https://medium.com/@sardorabdirayimov/lower-and-upper-mask-range-for-color-detection-opencv-7f64a5df568b
# AI (Claude) koristen za: objasnjenje koda i prilagodjavanje za rad sa slikom
# (slika umjesto kamere, H ogranicen na 179, gornje granice pocinju od maksimuma,
#  ispis pronadjenih granica, smanjenje slike za prikaz)
#
# Pomocni alat za pronalazenje HSV granica - nije dio rjesenja.
# Pokretanje: python hsv_trackbar.py data/images/image_1.jpg
# Izlaz: pritisnuti 'q' dok je prozor aktivan -> granice se ispisu u terminalu

import sys
import cv2 as cv
import numpy as np

Winname = "Frame:"

def nothing(x):
    # createTrackbar zahtijeva funkciju koja se poziva pri pomjeranju klizaca;
    # vrijednosti citamo sami u petlji, pa je prazna
    pass

# putanja do slike iz terminala, a ako nije zadata, koristi se image_1
putanja = sys.argv[1] if len(sys.argv) > 1 else 'data/images/image_1.jpg'
img = cv.imread(putanja)
img = cv.resize(img, None, fx=0.6, fy=0.6)     # samo za prikaz, da stane na ekran
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)       # imread daje BGR; slika je ista, pa jednom prije petlje

cv.namedWindow(Winname)
# H, S, V su donje granice; H2, S2, V2 gornje
# H u OpenCV-u ide do 179, S i V do 255
# gornje granice pocinju od maksimuma -> na pocetku sve prolazi, pa se suzava
cv.createTrackbar('H',  Winname, 0,   179, nothing)
cv.createTrackbar('S',  Winname, 0,   255, nothing)
cv.createTrackbar('V',  Winname, 0,   255, nothing)
cv.createTrackbar('H2', Winname, 179, 179, nothing)
cv.createTrackbar('S2', Winname, 255, 255, nothing)
cv.createTrackbar('V2', Winname, 255, 255, nothing)

while True:
    # citanje trenutnih pozicija klizaca
    H  = cv.getTrackbarPos('H',  Winname)
    S  = cv.getTrackbarPos('S',  Winname)
    V  = cv.getTrackbarPos('V',  Winname)
    H2 = cv.getTrackbarPos('H2', Winname)
    S2 = cv.getTrackbarPos('S2', Winname)
    V2 = cv.getTrackbarPos('V2', Winname)

    lower_boundary = np.array([H, S, V])
    upper_boundary = np.array([H2, S2, V2])
    mask = cv.inRange(hsv, lower_boundary, upper_boundary)   # binarna maska (0/255)
    final = cv.bitwise_and(img, img, mask=mask)              # samo pikseli koji su prosli, u boji

    cv.imshow(Winname, final)
    cv.imshow('Maska', mask)

    # ceka 30 ms na taster; 'q' = izlaz i ispis granica
    if cv.waitKey(30) == ord('q'):
        print('lower:', lower_boundary.tolist())
        print('upper:', upper_boundary.tolist())
        break

cv.destroyAllWindows()