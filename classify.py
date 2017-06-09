import numpy as np

try:
    import matplotlib.pyplot as plt
except:
    print("Błąd podczas importowania matplotlib")
    
from scipy.stats import norm

def najblizszaSrednia(ciagUczacy, obraz, klasy):

    srednieKlas = np.array([ciagUczacy[ciagUczacy[:, 1] == klasa][:, 0].mean() for klasa in klasy])

    odleglosci = abs(srednieKlas - obraz)
    
    return np.where(odleglosci == min(odleglosci))[0] + 1
    
def alfaNajblizszychSasiadow(alfa, ciagUczacy, obraz):

    odleglosci = np.array((abs(ciagUczacy[:, 0] - obraz), np.array(ciagUczacy[:, 1], dtype=int))).T
    
    alfaSasiadow = []
    for i in range(alfa):

        alfaSasiadow.append(odleglosci[odleglosci[:, 0] == min(odleglosci[:, 0])][0])
        odleglosci = odleglosci[odleglosci[:, 0] != min(odleglosci[:, 0])]

    alfaSasiadow = np.array(alfaSasiadow)
    
    if len(alfaSasiadow[alfaSasiadow[:, 1] == 1])\
       > alfa - len(alfaSasiadow[alfaSasiadow[:, 1] == 1]):

        return 1
    
    else:

        return 2

def Bayes(ciagUczacy,
          obraz,
          pstwaApriori,
          funkcjaGestosciRozkladu,
          parametryRozkladow):

    liczbaKlas = len(pstwaApriori)

    bezwarunkowaGestoscRozkladu = lambda x:\
                                  np.sum(np.array
                                         ([pstwaApriori[j]
                                           * funkcjaGestosciRozkladu
                                           (x,
                                            parametryRozkladow[j][1],
                                            parametryRozkladow[j][0])
                                           for j in range(liczbaKlas)]))
    
    pstwaAposteriori = [pstwaApriori[i]\
                        * funkcjaGestosciRozkladu\
                        (obraz,
                         parametryRozkladow[i][1],
                         parametryRozkladow[i][0])\
                        / bezwarunkowaGestoscRozkladu(obraz)\
                        for i in range(liczbaKlas)]

    return pstwaAposteriori.index(max(pstwaAposteriori)) + 1

if __name__ == "__main__":
 
    alfy = [1, 3, 5]
    klasy = [1, 2]
    polowaDlugosciCiaguUczacego = 1000
    polowaDlugosciCiaguTestowego = 50
    ciagUczacy = []
    ciagTestowy = []
    uzyjRozkladuNormalnego = True

    p1 = 0.5
    p2 = 0.5
    
    a1 = 5.5
    b1 = 9.5
    a2 = 6
    b2 = 10

    m1 = 0
    s1 = 2
    m2 = 7
    s2 = 2

    pstwaApriori = [p1, p2]
    
    parametryRozkladow = [[s1, m1], [s2, m2]] if uzyjRozkladuNormalnego\
                        else [[b1-a1, a1], [b2-a2, a2]]

    generator = np.random.randn if uzyjRozkladuNormalnego else np.random.rand
    
    funkcjaGestosciRozkladu = norm.pdf if uzyjRozkladuNormalnego else (lambda x, p1, p2: (1 / p2 if p1 < x < p1 + p2 else 0))
    
    for i in range(len(klasy)):

        ciagUczacy += [(parametryRozkladow[i][0] * generator() + parametryRozkladow[i][1], klasy[i])\
                       for j in range(polowaDlugosciCiaguUczacego)]
    
        ciagTestowy += [(parametryRozkladow[i][0] * generator() + parametryRozkladow[i][1], klasy[i])\
                        for j in range(polowaDlugosciCiaguTestowego)]

    ciagUczacy = np.array(ciagUczacy)
    ciagTestowy = np.array(ciagTestowy)
    
    iloscBlednychKlasyfikacjiNM = 0
    iloscBlednychKlasyfikacjiAlfaNN = 0
    iloscBlednychKlasyfikacjiBayes = 0
        
    for i in range(2 * polowaDlugosciCiaguTestowego):

        obraz = ciagTestowy[i][0]
        rzeczywistaKlasa = ciagTestowy[i][1]
        
        if najblizszaSrednia(ciagUczacy, obraz, klasy) != rzeczywistaKlasa:

            iloscBlednychKlasyfikacjiNM += 1
        
        if alfaNajblizszychSasiadow(alfy[2], ciagUczacy, obraz) != rzeczywistaKlasa:

            iloscBlednychKlasyfikacjiAlfaNN += 1
        
        if Bayes(ciagUczacy, obraz, pstwaApriori, funkcjaGestosciRozkladu, parametryRozkladow) != rzeczywistaKlasa:

            iloscBlednychKlasyfikacjiBayes += 1

    czestotliowoscBlednychKlasyfikacjiNM\
        = iloscBlednychKlasyfikacjiNM / 2 / polowaDlugosciCiaguTestowego
    
    czestotliowoscBlednychKlasyfikacjiAlfaNN\
        = iloscBlednychKlasyfikacjiAlfaNN / 2 / polowaDlugosciCiaguTestowego

    czestotliowoscBlednychKlasyfikacjiBayes\
        = iloscBlednychKlasyfikacjiBayes / 2 / polowaDlugosciCiaguTestowego

    print("NM: " + str(czestotliowoscBlednychKlasyfikacjiNM))
    print("k-NN: " + str(czestotliowoscBlednychKlasyfikacjiAlfaNN))
    print("Bayes: " + str(czestotliowoscBlednychKlasyfikacjiBayes))
