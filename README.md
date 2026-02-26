# Klasifikacija stanja u igri Dark Souls I

**Autor:** Filip Tot (SV14/2022)  
**Predmet:** Soft Computing

## 1. Definicija problema
Cilj projekta je razvoj modela za klasifikaciju stanja u video igri "Dark Souls I". Model na osnovu ulaznog frejma određuje u kojoj situaciji se trenutno nalazi igrač. Situacija će se klasifikovati u jednu od 4 definisane klase.

### Definisane klase:
1.  Exploration - generalno kretanje kroz igru
2.  Boss Fight - borba sa boss neprijateljima
3.  Menu/Inventory - prisustvo menija
4.  Death Screen - ekran koji se pojavi nakon što igrač pogine


## 2. Skup podataka
Podaci za treniranje i validaciju su prikupljeni iz javno dostupnih playthrough video snimaka. Iz snimaka su izvučeni frejmovi na određenim intervalima, i zatim ručno sortirani u odgovarajuće klase. Skup se sastoji od 600 slika, 150 po klasi.


## 3. Metodologija
Primenjena je tehnika Transfer Learning-a nad arhitekturom ResNet50 pre-treniranoj na ImageNet skupu.

### Arhitektura Modela:
1.  Base Model: ResNet50 (bez gornjih slojeva), korišćen kao ekstraktor obeležja.
2.  Head (Klasifikator): `GlobalAveragePooling2D` -> `Dropout(0.4)` -> `Dense(4, Softmax)`.

### Treniranje u dve faze:
Proces obučavanja je podeljen u dve faze radi stabilnosti:
1.  Faza 1: Zamrznuti su niži slojevi modela, radi treniranja klasifikatora. Koristi se veća brzina učenja (1e-3). Cilj je inicijalizacija težina nove Dense klase.
2.  Faza 2 (Fine-Tuning): Odmrznuti su niži slojevi modela (osim prvih 100 slojeva koji čuvaju osnovne vizuelne karakteristike). Treniran se cela mreže sa manjom brzinom učenja (1e-5) kako bi se model prilagodio art style-u igre.
Vršena je augmentacija (rotate, zoom, shift) u realnom vremenu kako bi se izbegao overfitting.


## 4. Rezultati i zaključak
Evaluacija je izvršena na validacionom skupu (20% podataka).

### Performanse:
*   Training Accuracy: ~98%
*   Validation Accuracy: ~86% - 94% 

### Analiza grešaka preko matrice konfuzije:
*   Death / Menu: Model postiže potpunu preciznost.
*   Boss vs Exploration: Model pravi male greške pri razlikovanju ove dve klase. Ovo je očekivano jer su vizuelno slični, i jedino se razlikuju po "health bar-u boss-a" na dnu ekrana, koji može biti zaklonjen tokom augmentacije.


## 5. Pokretanje koda

### Instalacija potrebnih modula
```bash
pip install -r requirements.txt
```

### Pokretanje treninga
Skripta automatski preuzima ResNet50 težine i pokreće treniranje:
```bash
python src/train_model.py
```

Za evaluaciju postojećeg modela bez treniranja:
```bash
python src/train_model.py --evaluate
```


## 6. Literatura i reference
1.  **Transfer learning guide:** [TensorFlow Core - Transfer learning & fine-tuning](https://www.tensorflow.org/guide/keras/transfer_learning)
2.  **ResNet arhitektura:** He, K., et al. "Deep Residual Learning for Image Recognition." [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
3.  **Data augmentation:** [Keras ImageDataGenerator Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)