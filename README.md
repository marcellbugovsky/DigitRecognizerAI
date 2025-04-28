# DigitRecognizerAI

Ein Projekt zur Erkennung handgeschriebener Ziffern mittels eines Multi-Layer Perceptron (MLP) Modells, implementiert in Python mit PyTorch. Das Projekt beinhaltet Funktionalitäten für das Training auf MNIST, das Fine-Tuning mit benutzerdefinierten Daten und eine grafische Benutzeroberfläche (GUI) zum Zeichnen, Vorhersagen und Sammeln neuer Trainingsdaten.

## Projektbeschreibung

Dieses Projekt implementiert ein End-to-End-System für die Erkennung handgeschriebener Ziffern:

1.  **Training:** Ein MLP-Modell wird auf dem MNIST-Datensatz trainiert (optional unter Einbeziehung eigener Daten).
2.  **Fine-Tuning:** Das vortrainierte Modell kann auf einem kleineren Datensatz benutzerdefinierter, über die GUI gesammelter Ziffern nachjustiert werden.
3.  **GUI:** Eine Tkinter-basierte Oberfläche ermöglicht das Zeichnen von Ziffern, das Erhalten von Modellvorhersagen und das Speichern der gezeichneten Ziffern mit korrekten Labels, um den Datensatz für das Fine-Tuning zu erweitern.

## Features

* **Modell:** Konfigurierbares Multi-Layer Perceptron (MLP) mit Optionen für Hidden Layer Größen, Dropout und Batch Normalization (`src/models/mlp.py`).
* **Datenverarbeitung:** Lädt MNIST und benutzerdefinierte Bilddaten, wendet Transformationen an und nutzt PyTorch DataLoaders (`src/data/`). Optionales Kombinieren von MNIST und benutzerdefinierten Daten für das initiale Training.
* **Training & Fine-Tuning:** Implementiert Trainings- und Evaluationsschleifen, unterstützt verschiedene Optimizer (Adam, SGD) und Learning Rate Scheduler (StepLR), speichert das beste Modell basierend auf Validierungsgenauigkeit (`src/training/trainer.py`, `main_train.py`, `main_finetune.py`).
* **Konfiguration:** Zentrale Konfiguration aller wichtigen Parameter über `config/config.yaml` (Pfade, Modellarchitektur, Trainings-/Fine-Tuning-Hyperparameter, GUI-Einstellungen).
* **Grafische Benutzeroberfläche (GUI):** Ermöglicht interaktives Zeichnen, Vorhersage durch das (optional feinjustierte) Modell und das Speichern von gezeichneten Ziffern mit korrekten/korrigierten Labels zur Datensammlung (`src/gui/`, `run_gui.py`).
* **Geräteunabhängigkeit:** Nutzt automatisch CUDA (GPU), falls verfügbar, sonst CPU (`src/utils/device.py`).

## Verwendete Technologien

* **Sprache:** Python 3
* **Kernbibliotheken:**
    * PyTorch (`torch`, `torchvision`)
    * Tkinter (für die GUI)
    * PyYAML (für die Konfiguration)
    * Pillow (PIL) (für Bildverarbeitung in der GUI)
    * NumPy (implizit durch PyTorch)

## Setup & Installation

1.  **Repository klonen:**
    ```bash
    git clone [https://github.com/marcellbugovsky/DigitRecognizerAI.git](https://github.com/marcellbugovsky/DigitRecognizerAI.git)
    cd DigitRecognizerAI
    ```
2.  **Virtuelle Umgebung erstellen (empfohlen):**
    ```bash
    python -m venv venv
    venv\Scripts\activate    # Windows
    ```
3.  **Abhängigkeiten installieren:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Konfiguration anpassen:**
    * Öffne `config/config.yaml` und passe bei Bedarf Pfade (`data_root`, `custom_data_root`, `model_save_dir`) und andere Einstellungen an. Stelle sicher, dass die Verzeichnisse existieren oder erstellt werden können.

## Verwendung

**Wichtiger Hinweis:** Führe die Skripte aus dem Hauptverzeichnis des Projekts (`DigitRecognizerAI/`) aus, damit die relativen Pfade zur `config.yaml` und den `src`-Modulen korrekt aufgelöst werden.

1.  **Initiales Training:**
    * Trainiert das MLP-Modell auf MNIST (oder kombinierten Daten, je nach `config.yaml`).
    * Speichert das trainierte Modell in dem in `config.yaml` definierten `model_save_dir`.
    ```bash
    python src/main_train.py
    ```

2.  **Grafische Benutzeroberfläche (GUI) starten:**
    * Lädt das trainierte (oder feinjustierte) Modell (gemäß `gui.model_suffix_to_load` in `config.yaml`).
    * Ermöglicht das Zeichnen von Ziffern und das Erhalten von Vorhersagen.
    * Ermöglicht das Speichern der gezeichneten Ziffern für das Fine-Tuning.
    ```bash
    python src/run_gui.py
    ```

3.  **Fine-Tuning (optional):**
    * Nachdem über die GUI eigene Ziffern gesammelt und gespeichert wurden (im `custom_data_root`-Verzeichnis).
    * Lädt das initial trainierte Modell und trainiert es auf den benutzerdefinierten Daten weiter.
    * Speichert das feinjustierte Modell unter einem neuen Namen (mit Suffix aus `config.yaml`).
    ```bash
    python src/main_finetune.py
    ```

## Konfiguration (`config/config.yaml`)

Die `config.yaml`-Datei steuert alle Aspekte des Projekts:

* **Pfade:** Speicherorte für Daten und Modelle.
* **Modellarchitektur:** Anzahl und Größe der Hidden Layer, Dropout-Rate, Verwendung von BatchNorm.
* **Datenlader:** Batch-Größe, Anzahl der Worker für paralleles Laden.
* **Initiales Training:** Anzahl Epochen, Lernrate, Optimizer, Scheduler-Einstellungen, Option zur Einbindung eigener Daten.
* **Fine-Tuning:** Eigene Hyperparameter (Epochen, Lernrate, Batch-Größe), Suffix für das gespeicherte Modell.
* **GUI:** Welches Modell geladen wird (initial oder feinjustiert), Fenstertitel, Canvas-Größe, Pinselgröße, Temperatur für die Konfidenzausgabe.

## Eigene Daten sammeln mit der GUI

1.  Starte die GUI (`python src/run_gui.py`).
2.  Zeichne eine Ziffer (0-9) in das weiße Feld.
3.  Klicke auf "Predict". Das Modell gibt eine Vorhersage aus.
4.  **Wenn die Vorhersage korrekt ist:** Klicke auf "Confirm ✔". Das Bild wird im entsprechenden Unterordner (z.B. `data/custom_digits/5/`) gespeichert.
5.  **Wenn die Vorhersage falsch ist:** Klicke auf "Correct ✘". Gib die korrekte Ziffer ein. Das Bild wird im richtigen Unterordner gespeichert.
6.  Klicke auf "Clear", um eine neue Ziffer zu zeichnen.
7.  Diese gesammelten Daten können dann für das Fine-Tuning (`python src/main_finetune.py`) verwendet werden.

## Lizenz

* Dieses Projekt steht unter der MIT-Lizenz.