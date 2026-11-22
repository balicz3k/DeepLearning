# Lab 6 – modele sekwencyjne i attention: materiał do nauki i obrony

Notatnik: `06_pytorch_moje.ipynb` (PyTorch, znakowy model językowy na „Winnie-the-Pooh”, RTX 4060, cały notatnik liczy się poniżej minuty).

Jak czytać: **A** – pojęcia, **B** – kod komórka po komórce, **C** – wyniki, **D** – pytania prowadzącego.

---

## A. Pojęcia

**Znakowy model językowy.** Model dostaje 80 kolejnych znaków tekstu i ma przewidzieć 81. znak. To klasyfikacja na tyle klas, ile jest różnych znaków (tu 53). Generowanie tekstu to powtarzanie tej predykcji: przewidziany znak dokleja się do wejścia i przewiduje następny (**autoregresja**).

**Słownik i kodowanie.** Każdy znak dostaje numer (`char_to_id`), a tekst staje się tablicą liczb. `id_to_char` zamienia numery z powrotem na znaki.

**Embedding.** `nn.Embedding(53, 48)` to tabela 53 wektorów po 48 liczb, uczonych razem z modelem. Numer znaku wybiera wiersz. Działa jak one-hot pomnożony przez macierz, ale bez tworzenia wektorów one-hot.

**RNN.** Sieć czyta sekwencję po jednym znaku i trzyma **stan ukryty** h (tu 96 liczb): $h_t = \tanh(W x_t + U h_{t-1} + b)$. Stan po ostatnim znaku podsumowuje całą sekwencję i z niego warstwa `Linear` liczy wynik dla każdego znaku. Problem: przy długich sekwencjach gradient przechodzi przez wiele mnożeń przez U i zanika, więc RNN słabo pamięta odległe znaki.

**LSTM i GRU (modele z bramkami).** Bramki to małe warstwy z sigmoidą (wartości 0–1), które decydują, ile starej informacji zachować, a ile nowej dopisać. **LSTM** ma osobną „komórkę pamięci” i 3 bramki (zapominania, wejścia, wyjścia). **GRU** jest prostszy: 2 bramki (aktualizacji i resetu), bez osobnej komórki, mniej parametrów (49 733 wobec 63 749 dla LSTM w tym notatniku). Bramki pozwalają gradientowi płynąć dalej bez zanikania.

**Cross-entropy i perplexity.** Loss to średnie $-\ln p(\text{prawdziwy znak})$. **Perplexity** = $e^{\text{loss}}$: model jest średnio tak niepewny, jakby losował równomiernie spośród tylu znaków. Perplexity 1 = pewność, 53 = zgadywanie.

**Temperatura i top-k.** Przed softmaxem logity dzieli się przez temperaturę T: T < 1 wyostrza rozkład (pewniejsze, powtarzalne wybory), T > 1 spłaszcza (więcej losowości). Top-k zostawia tylko k najbardziej prawdopodobnych znaków i losuje wśród nich.

**Attention pooling.** Zamiast brać tylko ostatni stan, model liczy dla każdej pozycji t wynik $s_t$ (warstwa `Linear(96, 1)`), zamienia wyniki na wagi softmaxem po czasie ($\sum_t w_t = 1$) i bierze ważoną sumę stanów: $c = \sum_t w_t h_t$. Wagi mówią, na które pozycje model „patrzy”.

**Podział danych w kolejności tekstu.** Przy tekście nie miesza się przykładów przed podziałem: walidacja to końcówka książki. Okna nachodzą na siebie (okno 80, przesunięcie 3), więc między treningiem a walidacją pomija się 27 przykładów, żeby żaden fragment tekstu walidacji nie trafił do treningu.

---

## B. Kod komórka po komórce

### Konfiguracja (z szablonu)
Stałe: okno 80 znaków, przesunięcie 3, embedding 48, stan ukryty 96, batch 128. Na GPU używam wariantu, który szablon sam proponuje: `MAX_CHARS = 160000`, `MAX_EXAMPLES = 30000`, 8 epok (przy domyślnych 5 epokach modele uczyły się głównie spacji i słowa „the”).

### Zadanie 1a – wczytanie i czyszczenie
- `strip_gutenberg` wycina licencję Project Gutenberg przed znacznikiem `*** START OF` i po `*** END OF`.
- `clean_text`: małe litery, typograficzne cudzysłowy i myślniki → ASCII, usunięcie `_` (kursywa), każdy ciąg białych znaków → jedna spacja.
- Korpus: tylko „Winnie-the-Pooh” (123 609 znaków), żeby walidacja miała ten sam styl co trening.
- Teksty pobrałem z mirrora `gutenberg.pglaf.org`, bo gutenberg.org miał przerwę techniczną, a link `pg11` z szablonu to „Alicja w Krainie Czarów”.

### Zadanie 1b – słownik
`sorted(set(raw_text))` → 53 znaki; `encoded` to tablica numerów znaków.

### Zadanie 1c – przykłady i podział
`make_examples`: okno zaczyna się co 3 znaki, X = 80 znaków, y = znak po oknie; limit 30 000 przykładów (pokrywają pierwsze 90 080 znaków). Podział 90/10 w kolejności, z przerwą 27 przykładów. DataLoader treningowy miesza batche (z ustalonym ziarnem), walidacyjny nie.

### Funkcje pomocnicze
- `run_epoch` – jedna epoka; `model.train(is_train)` włącza dropout tylko w treningu.
- `train_torch_model` – Adam (lr 0,001), `CrossEntropyLoss`, zapamiętuje wagi z epoki o najniższym validation loss.
- `perplexity(loss) = exp(loss)`.
- Model odniesienia: zawsze rozkład częstości znaków z treningu (val_loss 2,985, perplexity 19,79).

### Zadanie 2 – `RecurrentLM`
`Embedding → RNN/LSTM/GRU (batch_first) → ostatni stan outputs[:, -1, :] → Dropout(0,2) → Linear(96, 53)`. Model zwraca logity, softmax jest w `CrossEntropyLoss`.

### Zadanie 3 – LSTM, GRU i generowanie
- Ta sama klasa z `kind="lstm"` / `"gru"`, ta sama liczba epok dla wszystkich modeli (uczciwe porównanie).
- `sample_from_logits`: dzielenie przez temperaturę → maska top-k (`-inf` dla reszty) → softmax → `torch.multinomial` z ustalonym generatorem.
- `encode_seed`: znaki spoza słownika pomija, krótszy seed dopełnia spacjami z lewej.
- `generate_text`: pętla 300 razy: ostatnie 80 znaków → model → wylosowany znak → doklejenie do kontekstu.
- Seed: ostatnie 60 znaków pierwszego przykładu walidacyjnego (tekst, którego model nie widział).

### Zadanie 4 – attention
- `AttentionGRULM`: GRU zwraca wszystkie stany (N, 80, 96) → `Linear(96, 1)` daje wynik na pozycję → softmax po osi czasu → ważona suma → Dropout → Linear. Z `return_attention=True` zwraca też wagi.
- `AttentionLastGRULM` (mój dodatek do porównania): to samo, ale do warstwy wyjściowej idzie [kontekst, ostatni stan] – 192 liczby.
- 4b: dla jednej próbki – miejsce prawdziwego znaku w rankingu, 5 najbardziej prawdopodobnych znaków, wykres wag nad znakami kontekstu; potem średnie wagi na całej walidacji i próbki tekstu z trzech modeli przy tym samym ziarnie losowania.

---

## C. Wyniki (walidacja, najlepsza epoka = 8. dla wszystkich)

| model | parametry | val_loss | perplexity | val_acc |
|---|---:|---:|---:|---:|
| częstości znaków | – | 2,985 | 19,79 | 0,193 |
| RNN | 21 701 | 1,780 | 5,93 | 0,509 |
| LSTM | 63 749 | 1,748 | 5,74 | 0,497 |
| GRU | 49 733 | **1,669** | **5,31** | **0,524** |
| GRU+attention | 49 830 | 2,441 | 11,49 | 0,326 |
| GRU+attention+ostatni stan | 54 918 | 1,661 | 5,27 | 0,521 |

- GRU najlepszy z modeli rekurencyjnych; różnice niewielkie przy 8 epokach.
- Czyste attention przez 6 epok stoi na poziomie modelu częstości znaków i rusza dopiero w 7.–8. epoce.
- Po treningu attention patrzy prawie tylko na koniec sekwencji: średnio ostatnie 10 pozycji ma 98,4% wagi.
- Generowany tekst: poprawne spacje, interpunkcja, imiona („pooh”, „piglet”), częste słowa; brak sensu zdań. Niska temperatura → powtórzenia, wysoka → zmyślone słowa.

---

## D. Pytania, które może zadać prowadzący

1. **Czemu ostatnia warstwa nie ma softmaxu?** `CrossEntropyLoss` przyjmuje logity i sam liczy log-softmax (stabilniej numerycznie).
2. **Co znaczy perplexity 5,3?** Model jest średnio tak niepewny, jakby wybierał równomiernie spośród ok. 5 znaków; model częstości – spośród ok. 20.
3. **Czemu nie mieszasz danych przed podziałem na train/val?** Sąsiednie okna nachodzą na siebie; po wymieszaniu walidacja zawierałaby prawie te same fragmenty co trening i wynik byłby zawyżony. Dlatego walidacja to koniec tekstu plus przerwa 27 przykładów.
4. **Czym różni się LSTM od GRU?** LSTM: komórka pamięci + 3 bramki; GRU: 2 bramki, stan i pamięć w jednym wektorze, mniej parametrów. Oba ograniczają zanikanie gradientu.
5. **Czemu LSTM ma niższy loss od RNN, ale niższe accuracy?** Loss ocenia całe rozkłady (pewność dobrych odpowiedzi), accuracy tylko trafienie najwyżej ocenionego znaku.
6. **Jak działa temperatura?** Dzieli logity przed softmaxem; T < 1 wzmacnia różnice (powtarzalny tekst), T > 1 je zmniejsza (więcej losowości i błędów).
7. **Czemu czyste attention uczy się tak wolno?** Na starcie wagi są równe (1/80), więc kontekst to średnia wszystkich stanów, w której ginie informacja o ostatnich znakach; gradient na każdą pozycję jest 80 razy mniejszy. Model musi najpierw nauczyć się, gdzie patrzeć.
8. **Co pokazują wagi attention?** Po treningu prawie cała waga jest na ostatnich znakach – dla predykcji następnego znaku liczy się głównie bieżące słowo. To samo daje ostatni stan GRU, dlatego attention nie poprawia wyniku.
9. **Czy attention w tym notatniku to transformer?** Nie. To jedna warstwa ważonej sumy stanów GRU (attention pooling), bez zapytań/kluczy/wartości dla każdej pozycji i bez wielu głów.
10. **Po co ustalone ziarno w losowaniu znaków?** Żeby porównanie tekstów z GRU i GRU+attention różniło się tylko modelem, a nie losowaniem.
11. **Dlaczego wariant GPU, a nie domyślne ustawienia?** Szablon proponuje go wprost; trening trwa kilka sekund, a przy 5 epokach modele nie zdążyły nauczyć się nawet częstych słów.
12. **Czemu dropout 0,2?** Typowa wartość, nie stroiłem jej; train loss i val loss są bliskie, więc przeuczenia nie ma.
