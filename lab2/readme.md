# Lab 2 – Neuron, MLP i backpropagation – raport

Rozwiązanie znajduje się w `02_pytorch_rozwiazanie.ipynb`. Oryginalny notatnik `02_pytorch.ipynb` nie był zmieniany. W kopii zostawiłem wszystkie komentarze `# TODO`, żeby było widać, jakie było polecenie. Kod jest dopisany bezpośrednio pod nimi.

## Wyniki w skrócie

| Model | accuracy train | accuracy validation | accuracy test |
|---|---|---|---|
| Pojedynczy neuron (NumPy) | 0.868 | 0.825 | 0.920 |
| MLP 2-8-1, tanh (NumPy) | 0.978 | 0.950 | 0.980 |
| MLP 2-16-1, tanh (PyTorch, Adam) | – | 0.960 | 0.985 |

Sprawdzenie gradientu metodą różnic skończonych dla `W1[0, 0]`: gradient analityczny i numeryczny to w obu przypadkach `-0.0002211708`, a błąd względny wynosi `2.6e-09`. Oznacza to, że ręcznie napisana propagacja wsteczna jest poprawna.

---

## Ćwiczenie 1 – dane

### Co robimy

`make_moons` generuje dwa „półksiężyce” wchodzące jeden w drugi. Każdy punkt ma 2 cechy (`x1`, `x2`) i etykietę 0 albo 1. Parametr `noise=0.2` dodaje szum, więc klasy trochę na siebie nachodzą. Takiego zbioru **nie da się rozdzielić prostą linią**, i właśnie dlatego się go tu używa: widać na nim różnicę między modelem liniowym a nieliniowym.

Wygenerowałem 1000 punktów i podzieliłem je dwoma wywołaniami `train_test_split`:

1. najpierw odcinam 20% na test (200 punktów),
2. z pozostałych 800 odcinam 25% na walidację (200 punktów), a reszta to train (600 punktów).

Proporcje wychodzą 60/20/20.

`stratify=y` pilnuje, żeby w każdym zbiorze był taki sam udział obu klas jak w całych danych. Bez tego przy pechowym losowaniu w jednym zbiorze mogłoby być wyraźnie więcej jedynek niż zer.

### Po co trzy zbiory

- **train** – na nim model uczy się wag,
- **validation** – na nim porównujemy modele i dobieramy ustawienia (learning rate, liczbę neuronów itp.),
- **test** – używamy go dopiero na samym końcu, raz, żeby uczciwie ocenić wybrany model. Gdybyśmy dobierali coś pod test, wynik na teście przestałby być uczciwą oceną.

### Skalowanie

`StandardScaler` przesuwa każdą cechę tak, żeby miała średnią 0, i skaluje ją do odchylenia standardowego 1. Ułatwia to gradient descent, bo wszystkie cechy mają podobną skalę.

Ważne jest, żeby **`fit` robić tylko na train** (`fit_transform(X_train_raw)`), a na validation i test robić już wyłącznie `transform`. Średnia i odchylenie liczone są wtedy tylko z danych treningowych. Gdybyśmy dopasowali skaler na wszystkich danych, informacja ze zbioru testowego „wyciekłaby” do treningu (tzw. *data leakage*).

![dane](raport_img/dane.png)

---

## Ćwiczenie 2 – pojedynczy neuron logistyczny

### Sigmoid

\[ \sigma(z) = \frac{1}{1 + e^{-z}} \]

Sigmoid zamienia dowolną liczbę na wartość z przedziału (0, 1). Interpretujemy ją jako prawdopodobieństwo klasy 1.

### Binary crossentropy (BCE)

\[ L = -\frac{1}{m}\sum_{i=1}^{m}\Big[y_i \log a_i + (1-y_i)\log(1-a_i)\Big] \]

Jeśli prawdziwa klasa to 1, liczy się tylko \(-\log a\): kara jest mała, gdy model daje \(a\) bliskie 1, i rośnie do nieskończoności, gdy \(a \to 0\). Dla klasy 0 działa to symetrycznie. Wynik to średnia po wszystkich \(m\) przykładach.

`np.clip(y_score, eps, 1 - eps)` chroni przed `log(0)`, które dałoby `-inf`.

### Model i trening

Neuron liczy \(z = XW + b\), a potem \(a = \sigma(z)\). Przy dwóch cechach `W` ma kształt `(2, 1)`, a `b` to jedna liczba.

Gradient descent w każdej epoce:

1. **forward**: liczymy `z`, `a` i stratę,
2. **backward**: liczymy pochodne straty po wagach,
3. **update**: `W -= learning_rate * dW`, czyli robimy mały krok w kierunku, w którym strata maleje.

Wzory `dZ = (a - y) / m`, `dW = X.T @ dZ`, `db = sum(dZ)` biorą się z tego, że dla połączenia sigmoid + BCE pochodna po \(z\) upraszcza się do bardzo prostej postaci \(a - y\) (różnica między przewidywaniem a prawdą). Dzielenie przez \(m\) wynika ze średniej w BCE. `X.T @ dZ` sumuje wkład wszystkich przykładów do gradientu każdej wagi.

### Pomocnicze funkcje

- `accuracy_from_scores` – score ≥ 0.5 traktujemy jako klasę 1, a potem liczymy `accuracy_score`.
- `plot_decision_boundary` – tworzy gęstą siatkę 300×300 punktów, liczy na niej prawdopodobieństwo modelu i koloruje tło (niebieskie → klasa 0, czerwone → klasa 1). Czarna linia to poziom 0.5, czyli granica decyzyjna. Funkcja ma dodatkowy, opcjonalny argument `ax`, z którego korzysta ćwiczenie 5 do narysowania trzech wykresów obok siebie. Bez niego działa dokładnie tak, jak w szablonie.

### Wynik

![strata neuronu](raport_img/neuron_loss.png)
![granica neuronu](raport_img/neuron_granica.png)

Granica jest **prostą linią**. Tak musi być, bo warunek \(\sigma(XW+b) = 0.5\) jest równoważny \(XW + b = 0\), a to równanie prostej. Pojedynczy neuron nie umie wygiąć granicy wokół półksiężyców, dlatego zatrzymuje się na około 83% accuracy na walidacji.

---

## Ćwiczenie 3 – dwuwarstwowy MLP w NumPy

### Architektura

```
X (m×2) → Linear(2→8) → tanh → Linear(8→1) → sigmoid → a
```

- \(Z_1 = X W_1 + b_1\), \(A_1 = \tanh(Z_1)\) – warstwa ukryta z 8 neuronami,
- \(Z_2 = A_1 W_2 + b_2\), \(A_2 = \sigma(Z_2)\) – wyjście.

Nieliniowa funkcja `tanh` w środku jest kluczowa. Bez niej dwie warstwy liniowe złożyłyby się w jedną warstwę liniową i znowu dostalibyśmy prostą granicę.

### Inicjalizacja

Wagi losuję z rozkładu normalnego z odchyleniem \(1/\sqrt{n_{in}}\) (inicjalizacja w stylu Xaviera), a biasy ustawiam na zera. Wagi muszą być losowe, bo gdyby wszystkie były równe, każdy neuron ukryty liczyłby to samo i dostawałby taki sam gradient, więc nigdy by się nie zróżnicowały. Skala \(1/\sqrt{n_{in}}\) sprawia, że na starcie `tanh` nie jest nasycony (nie siedzi przy ±1, gdzie pochodna jest prawie zerowa).

### Cache

`mlp_forward` zwraca oprócz wyniku słownik `cache` z `X`, `A1` i `A2`. Propagacja wsteczna potrzebuje właśnie tych wartości, a zapamiętanie ich pozwala nie liczyć forwardu drugi raz.

### Backpropagation krok po kroku

To jest reguła łańcuchowa stosowana od końca sieci do początku:

| krok | wzór | skąd |
|---|---|---|
| `dZ2` | \((A_2 - y)/m\) | sigmoid + BCE, tak samo jak w ćw. 2 |
| `dW2` | \(A_1^T \, dZ_2\) | \(Z_2 = A_1 W_2 + b_2\), więc wejściem tej warstwy jest \(A_1\) |
| `db2` | suma `dZ2` po przykładach | bias jest dodawany do każdego przykładu |
| `dA1` | \(dZ_2 \, W_2^T\) | gradient „cofa się” przez wagi do warstwy ukrytej |
| `dZ1` | \(dA_1 \odot (1 - A_1^2)\) | pochodna tanh to \(1 - \tanh^2\) (mnożenie element po elemencie) |
| `dW1` | \(X^T \, dZ_1\) | jak dla `dW2`, tylko wejściem jest \(X\) |
| `db1` | suma `dZ1` po przykładach | |

`update_params` przechodzi po wszystkich parametrach i wykonuje `param -= lr * grad`.

### Sprawdzenie gradientu (gradient check)

Pochodną można przybliżyć bez żadnych wzorów:

\[ \frac{\partial L}{\partial w} \approx \frac{L(w + \varepsilon) - L(w - \varepsilon)}{2\varepsilon} \]

Biorę jedną wagę `W1[0, 0]`, zwiększam ją o \(\varepsilon = 10^{-5}\), liczę stratę, potem zmniejszam o \(\varepsilon\), znowu liczę stratę i przywracam oryginalną wartość. Wynik porównuję z gradientem z `mlp_backward` za pomocą błędu względnego \(|g_a - g_n| / (|g_a| + |g_n|)\). Wartości rzędu \(10^{-7}\) i mniejsze oznaczają, że backprop jest dobrze napisany. Tutaj wyszło \(2.6 \cdot 10^{-9}\).

### Wynik

![strata MLP](raport_img/mlp_loss.png)
![granica MLP](raport_img/mlp_granica.png)

Granica jest wygięta i podąża za kształtem półksiężyców. Accuracy na walidacji rośnie z 0.825 do 0.95.

---

## Ćwiczenie 4 – ten sam pomysł w PyTorch

### Model

```python
nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1))
```

Na końcu **nie ma sigmoidu**. Model zwraca *logity*, czyli surowe \(z\). Funkcja `nn.BCEWithLogitsLoss` sama nakłada sigmoid i liczy BCE w jednym, numerycznie stabilnym kroku, więc nie trzeba niczego przycinać jak `np.clip` w wersji NumPy. Sigmoid nakładamy dopiero przy predykcji.

### Optymalizator

Zamiast zwykłego gradient descent używam `Adam` z `lr=0.03`. Adam dla każdego parametru osobno dobiera wielkość kroku na podstawie historii gradientów, dzięki czemu trening jest szybszy i stabilniejszy. Widać to na wykresie: strata spada gwałtownie już w pierwszych ~150 epokach.

### Pętla treningowa

Każda epoka to pięć kroków, które w NumPy pisaliśmy ręcznie:

1. `optimizer.zero_grad()` – zeruje gradienty z poprzedniej epoki. PyTorch domyślnie je **dodaje**, więc bez zerowania by się kumulowały.
2. `logits = torch_model(X_train_torch)` – forward pass.
3. `loss = loss_fn(logits, y_train_torch)` – strata.
4. `loss.backward()` – PyTorch automatycznie liczy wszystkie gradienty (autograd). To odpowiednik całej funkcji `mlp_backward`.
5. `optimizer.step()` – aktualizacja wag, odpowiednik `update_params`.

### Normy gradientów

Po treningu wypisuję normę gradientu każdego parametru (`param.grad.norm()`). Wszystkie są rzędu \(10^{-4}\), czyli bardzo małe. Oznacza to, że model doszedł w okolice minimum i dalszy trening niewiele by już zmienił. Jest to też szybki test, że gradienty w ogóle płyną do wszystkich warstw (żaden nie wynosi zero ani nie „eksplodował”).

![strata PyTorch](raport_img/torch_loss.png)

### Predykcja

`predict_torch_scores`:

- `model.eval()` przełącza model w tryb ewaluacji. Przy tej architekturze nic to nie zmienia, ale to dobry nawyk, bo warstwy takie jak Dropout czy BatchNorm zachowują się inaczej w treningu i przy predykcji,
- `torch.no_grad()` wyłącza śledzenie gradientów, bo przy predykcji nie są potrzebne. Jest szybciej i zużywa mniej pamięci,
- `torch.sigmoid(logits)` zamienia logity na prawdopodobieństwa, a `.numpy()` oddaje wynik jako tablicę NumPy, żeby działały te same funkcje co wcześniej.

![granica PyTorch](raport_img/torch_granica.png)

Accuracy na walidacji: PyTorch 0.96, NumPy 0.95. Wyniki są praktycznie takie same, co ma sens, bo to ten sam typ modelu. PyTorch ma trochę więcej neuronów (16 zamiast 8) i lepszy optymalizator.

---

## Ćwiczenie 5 – test i porównanie

Zbiór testowy jest użyty dopiero teraz, po obejrzeniu wyników walidacyjnych. Model PyTorch ma na teście accuracy **0.985**.

![porównanie](raport_img/porownanie.png)

- **Pojedynczy neuron** – prosta linia. Część punktów każdej klasy zawsze leży po złej stronie.
- **MLP NumPy** i **MLP PyTorch** – wygięta granica w kształcie litery S, która wchodzi między półksiężyce. Model PyTorch rzeczywiście nauczył się granicy nieliniowej, a nie czegoś w rodzaju perceptronu. Gdyby tak nie było, jego wykres wyglądałby jak pierwszy.

Accuracy na teście jest odrobinę wyższe niż na walidacji. To normalne przy 200 punktach: jeden punkt to 0.5 punktu procentowego, więc różnice rzędu 1–3% to w dużej mierze kwestia tego, które punkty trafiły do którego zbioru.

---

## Jak uruchomić

W `lab2/.venv` jest osobne środowisko z `torch`, `scikit-learn`, `matplotlib`, `numpy` i `ipykernel` (globalnie nic nie było instalowane). W Cursorze wystarczy otworzyć notatnik i wybrać kernel **„Python (lab2 .venv)”**, a potem *Run All*. Z terminala:

```bash
cd lab2
.venv/bin/jupyter nbconvert --to notebook --execute --inplace 02_pytorch_rozwiazanie.ipynb
```

Wyniki są powtarzalne, bo ziarna losowości są ustawione (`np.random.seed`, `torch.manual_seed`, `random_state`).
