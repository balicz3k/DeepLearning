# Lab 5 – CNN i transfer learning: ściąga do obrony

Notatnik: `05_pytorch_moje.ipynb` (PyTorch, CIFAR-10, RTX 4060, całość liczy się ok. 20 minut).

## Wyniki w jednej tabeli

| Model | Trenowane parametry | train_acc | val_acc | Najlepsza epoka | Czas |
|---|---:|---:|---:|---:|---:|
| CNN od zera (baseline) | 620 810 | 0.928 | 0.784 | 11 z 15 | 267 s |
| CNN od zera + regularyzacja | 620 810 | 0.800 | 0.827 | 30 z 30 | 536 s |
| MobileNetV2 – zamrożony backbone | 12 810 | 0.812 | 0.826 | 3 z 5 | 134 s |
| MobileNetV2 – fine-tuning | 1 538 890 | 0.962 | **0.892** | 6 z 8 | 239 s |

Test (tylko najlepszy model według walidacji, liczony raz): **accuracy 0.890** na 10 000 obrazach. Najsłabsza klasa to cat (F1 0.788), najlepsze to ship (0.944) i automobile (0.935). Najczęstsza pomyłka: pies uznany za kota (151 obrazów).

---

## Zadanie 1 – dane i CNN od zera

**Dane.** CIFAR-10 ma 50 000 obrazów treningowych i 10 000 testowych, każdy 32x32 RGB, 10 klas. Z treningowych wydzielam 5000 na walidację z `stratify`, więc w walidacji jest po 500 obrazów każdej klasy. Test ruszam dopiero na samym końcu.

**Normalizacja.** Średnią i odchylenie każdego kanału RGB liczę tylko na 45 000 obrazów treningowych, żeby nie korzystać z informacji z walidacji. Wyszło średnio (0.491, 0.482, 0.446), odchylenie (0.247, 0.243, 0.262). `Normalize` odejmuje średnią i dzieli przez odchylenie, więc wejście ma wartości wokół zera. Sieci uczą się wtedy szybciej i stabilniej.

**Trzy różne zestawy transformacji:**
- **baseline:** tylko `ToTensor` i `Normalize`,
- **model regularyzowany:** dodatkowo losowe przesunięcie i odbicie (tylko na treningu),
- **transfer learning:** `Resize` do 96x96 i normalizacja statystykami ImageNet, bo takie dane widział MobileNetV2.

**Architektura `ScratchCNN`.** Trzy bloki: Conv 3x3 → BatchNorm → ReLU, a po każdym MaxPool 2x2.
- Rozmiar obrazu: 32 → 16 → 8 → 4.
- Liczba kanałów: 3 → 32 → 64 → 128.
- Klasyfikator: Flatten (128·4·4 = 2048) → Linear 256 → ReLU → Linear 10. Dropout stoi przed obiema warstwami Linear.
- `padding=1` przy filtrze 3x3 zachowuje rozmiar obrazu.
- `forward_features` zwraca aktywacje po każdym bloku, potrzebne do Zadania 2.

**Pętla treningowa.**
- `run_epoch` przełącza tryb `train`/`eval`, przenosi dane na GPU i liczy loss oraz accuracy. `backward` i `step` wykonuje tylko w treningu, a w ewaluacji wyłącza liczenie gradientów.
- `train_torch_model` trenuje model przez zadaną liczbę epok i zapamiętuje wagi z najlepszej epoki na walidacji (checkpoint). Na końcu przywraca te wagi.
- Optymalizator to AdamW. Przy `weight_decay=0` działa jak zwykły Adam.

**Wynik baseline.**
- Train dochodzi do 0.97, a val zatrzymuje się na ok. 0.78.
- Loss walidacyjny rośnie od 8. epoki (z 0.681 do 1.112), choć treningowy dalej spada. To klasyczne przeuczenie: model zapamiętuje obrazy treningowe zamiast uczyć się ogólnych cech.
- Checkpoint wybrał epokę 11 (val 0.784).

## Zadanie 2 – filtry i aktywacje

- **Filtry pierwszej warstwy** to tensor (32, 3, 3, 3): 32 filtry, każdy 3x3 na trzech kanałach RGB. Każdy filtr skaluję osobno do [0, 1] i rysuję jako mały kolorowy obrazek. Przy 3x3 nie widać wyraźnych wzorów. Każdy filtr to kontrast jasności lub koloru w jakimś kierunku, czyli detektor krawędzi albo plamy danej barwy.
- **Mapy aktywacji** pokazują, jak mocno każdy kanał reaguje na każde miejsce obrazu (jasne = silna reakcja po ReLU).
  - Blok 1 (32x32) wciąż przypomina konia. Różne kanały podświetlają sylwetkę, pionowe krawędzie nóg, górną krawędź grzbietu albo tło.
  - Blok 2 (16x16) ma rzadsze, mniej czytelne mapy, bo łączy cechy z bloku 1 w bardziej złożone wzory.
  - Im głębiej, tym mniejsza rozdzielczość i bardziej abstrakcyjne cechy.

## Zadanie 3 – poprawa generalizacji

Architektura jest ta sama co w baseline, więc różnica wynika tylko z regularyzacji. Zastosowane techniki:
1. **Augmentacja:** `RandomCrop(32, padding=4)` przesuwa obraz o najwyżej 4 piksele, `RandomHorizontalFlip` odbija go w poziomie. Klasa się nie zmienia, a model co epokę widzi trochę inne obrazy, więc trudniej mu je zapamiętać. Odbicia w pionie nie używam, bo daje nienaturalne obrazy.
2. **Dropout 0.3:** w treningu losowo wyłącza 30% neuronów, więc sieć nie może polegać na pojedynczych połączeniach. W ewaluacji (`model.eval()`) dropout jest wyłączony.
3. **Weight decay 0.05 (AdamW):** w każdym kroku lekko ściąga wagi do zera, czyli karze duże wagi. AdamW stosuje to bezpośrednio do wag, niezależnie od gradientu.

Wynik: val wzrosło z 0.784 do 0.827 (+4,4 pp), a różnica train–val zmalała z 14,4 pp do −2,8 pp. Train jest niższy od val, bo liczony jest na obrazach po augmentacji i z dropoutem. Na walidacji model widzi czyste obrazy bez dropoutu, więc ma łatwiej. Loss walidacyjny spada do końca, więc więcej epok mogłoby jeszcze pomóc.

## Zadanie 4 – transfer learning (MobileNetV2)

- **Skąd model:** MobileNetV2 nauczony na ImageNet (1,2 mln zdjęć, 1000 klas). Jego warstwy konwolucyjne (`features`) potrafią już wykrywać krawędzie, tekstury i części obiektów.
- **Etap 1, zamrożony backbone:** `requires_grad=False` dla całego `features` i nowa głowica Dropout 0.2 + Linear(1280 → 10). Trenuje się tylko 12 810 parametrów. Po 3 epokach val wynosi 0.826, czyli tyle co nasz najlepszy CNN po 30 epokach.
- **Pułapka z BatchNorm:** `model.train()` przełącza też warstwy BatchNorm w tryb treningowy, a wtedy nadpisują one swoje statystyki (running mean/var) statystykami z CIFAR, nawet gdy ich wagi są zamrożone. Funkcja `keep_frozen_batchnorm_in_eval` zostawia zamrożone BN w trybie eval, więc backbone działa tak, jak po treningu na ImageNet.
- **Etap 2, fine-tuning:** odblokowuję moduły `features[15:]`, czyli bloki 15–17 i końcowy Conv 1x1 (1,54 mln parametrów). Learning rate jest 10 razy mniejszy (1e-4). Val wzrasta do 0.892.
- **Przeuczenie przy fine-tuningu:** po kilku epokach loss walidacyjny rośnie (z 0.319 do 0.367), a train_acc dochodzi do 0.977. Checkpoint zachowuje epokę 6.
- **Rozmiar wejścia 96x96:** przy 32x32 po pięciu zmniejszeniach rozdzielczości w MobileNetV2 zostałby obraz 1x1 i sieć straciłaby informację przestrzenną. 96x96 to kompromis między jakością a czasem.

## Wynik końcowy

- Model wybieram tylko po `val_accuracy`. Najlepszy jest MobileNetV2 po fine-tuningu.
- Na teście ma 0.890, prawie tyle co na walidacji (0.892), więc wybór na walidacji nie był przeoptymalizowany.
- Macierz pomyłek: najwięcej błędów to pary podobnych klas: pies↔kot (151 + 69), samochód↔ciężarówka (44 + 32), a także ptak mylony z jeleniem, kotem i żabą. To klasy podobne kształtem i kolorem, trudne do odróżnienia przy małej rozdzielczości.

---

## 12 pytań, które może zadać prowadzący

1. **Po co zamrażać backbone?**
   Wagi z ImageNet są już dobre, a nowa głowica jest losowa. Gdybyśmy od razu trenowali wszystko, duże gradienty z losowej głowicy zepsułyby pretrenowane cechy. Do tego trening tylko głowicy jest szybki (12 tys. parametrów zamiast 2,2 mln) i mniej podatny na przeuczenie.

2. **Dlaczego mały learning rate przy fine-tuningu?**
   Odblokowane wagi są już blisko dobrego rozwiązania. Duży krok przesunąłby je daleko i model „zapomniałby” cech z ImageNet. Przy LR 1e-4 tylko lekko dopasowujemy je do CIFAR-10.

3. **Dlaczego odblokowujemy ostatnie, a nie pierwsze bloki?**
   Pierwsze warstwy wykrywają krawędzie i tekstury, przydatne dla każdych zdjęć. Ostatnie są najbardziej specyficzne dla klas ImageNet, więc to je warto dopasować do nowych klas.

4. **Jak czytać mapy aktywacji?**
   Każda mapa to odpowiedź jednego filtra w każdym miejscu obrazu (po ReLU, więc tylko wartości dodatnie). Jasne miejsca to te, gdzie występuje wzór, którego szuka filtr. W płytkich warstwach to krawędzie i plamy kolorów, w głębszych coraz bardziej złożone i mniej czytelne wzory przy niższej rozdzielczości.

5. **Skąd różnica train/val w baseline i co ona oznacza?**
   Model z 620 tys. parametrów i bez regularyzacji zapamiętuje 45 tys. obrazów treningowych (train 0.97) zamiast uczyć się ogólnych cech (val 0.78). Rosnący loss walidacyjny przy spadającym treningowym to sygnał przeuczenia.

6. **Dlaczego w modelu regularyzowanym train_acc jest niższe niż val_acc?**
   Train liczony jest w trybie treningowym: na obrazach po augmentacji i z aktywnym dropoutem, czyli na trudniejszych danych. Walidacja idzie w trybie `eval` na czystych obrazach. Nie oznacza to błędu ani wycieku.

7. **Co robi BatchNorm i czemu jest w baseline?**
   Normalizuje wyjście warstwy w obrębie batcha (średnia 0, wariancja 1), a potem skaluje je i przesuwa wyuczonymi parametrami. Stabilizuje i przyspiesza uczenie. Jest w obu modelach od zera, żeby porównanie w Zadaniu 3 pokazywało tylko efekt augmentacji, dropoutu i weight decay.

8. **Dlaczego w transfer learningu inna normalizacja i rozmiar 96x96?**
   Backbone był uczony na obrazach znormalizowanych statystykami ImageNet, więc dane muszą mieć taki sam rozkład. Rozmiar 96x96 jest potrzebny, bo przy 32x32 po pięciu zmniejszeniach rozdzielczości mapa cech miałaby 1x1 piksel.

9. **Po co wydzielać walidację, skoro jest test?**
   Walidacja służy do decyzji: wyboru epoki (checkpoint), hiperparametrów i najlepszego modelu. Test to niezależna ocena na końcu. Gdybyśmy wybierali na teście, wynik testowy byłby zawyżony.

10. **Co to jest checkpoint najlepszej epoki i czym różni się od early stoppingu?**
    Po każdej epoce, gdy val_acc jest najwyższe do tej pory, kopiuję `state_dict`, a na końcu przywracam te wagi. Early stopping dodatkowo przerywa trening po kilku epokach bez poprawy. Efekt jest podobny, a ja zachowuję pełne krzywe uczenia.

11. **Dlaczego model myli psy z kotami?**
    Przy 32x32 oba gatunki mają podobny kształt, pozę i kolory sierści, a cechy, które je różnią (pysk, uszy), zajmują kilka pikseli. Klasa cat ma najniższe F1 (0.788).

12. **Co dokładnie robi `RandomCrop(32, padding=4)`?**
    Dookoła obrazu dodaje 4-pikselową ramkę zer i wycina losowy fragment 32x32. Obraz jest więc przesunięty o 0–4 piksele w dowolną stronę. Uczy to model, że obiekt może być w różnych miejscach kadru (niezmienniczość na przesunięcie).

**Dodatkowe:**
- **Ile parametrów ma pierwsza warstwa?** 32·3·3·3 = 864 wagi + 32 biasy = 896.
- **Dlaczego AdamW, a nie Adam z `weight_decay`?** W Adamie kara L2 jest skalowana przez adaptacyjny krok, więc dla różnych wag działa różnie. AdamW odejmuje ją bezpośrednio od wag.
