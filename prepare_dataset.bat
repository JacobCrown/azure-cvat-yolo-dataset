@echo off
setlocal

REM === Konfiguracja Użytkownika ===
REM Zmień te ścieżki i nazwy zgodnie ze swoim projektem

REM Ścieżka do interpretera Python (jeśli nie jest w PATH lub używasz venv)
REM Zostaw pustą, jeśli 'python' działa globalnie: set PYTHON_EXE=
REM Przykład dla venv: set PYTHON_EXE=C:\path\to\your\venv\Scripts\python.exe
set PYTHON_EXE=python

REM Folder zawierający skrypty Python
set SCRIPTS_DIR=scripts

REM Nazwa kontenera w Azure Blob Storage
set CONTAINER_NAME=maxdrogi-yolo

REM Nazwy plików ZIP z adnotacjami CVAT (XML) - oddzielone spacją
set XML_ANNOTATION_BLOBS=cvat.zip cvat2.zip

REM Nazwy plików ZIP z adnotacjami YOLO (TXT + obj.names) - oddzielone spacją
set YOLO_ANNOTATION_BLOBS=yolo1.zip yolo2.zip

REM Nazwa folderu, który zostanie utworzony dla datasetu YOLO
set DATASET_DIR=yolo_dataset_final

REM Nazwa pośredniego pliku z listą obrazów do treningu
set TRAIN_LIST_FILE=to_train_list.txt

REM Nazwa pliku mapowania nazw (oryginalne Azure -> spłaszczone lokalne)
REM Ten plik zostanie utworzony WEWNĄTRZ folderu DATASET_DIR
set MAPPING_FILENAME=azure_to_local_map.json

REM Nazwa pliku z nazwami klas wewnątrz archiwów YOLO ZIP
set CLASS_NAMES_FILENAME=obj.names

REM (Opcjonalnie) Folder bazowy wewnątrz YOLO ZIP do pominięcia
set YOLO_ZIP_BASE_STRUCTURE=obj_train_data

REM === Koniec Konfiguracji Użytkownika ===

REM --- Krok 1: Znajdź obrazy do treningu na podstawie XML ---
echo.
echo === Krok 1: Wyszukiwanie obrazow do treningu z XML... ===
REM Uruchomienie skryptu w jednej linii dla pewności
%PYTHON_EXE% "%SCRIPTS_DIR%\find_images_to_train.py" --container-name "%CONTAINER_NAME%" --blob-names %XML_ANNOTATION_BLOBS% --output-file "%TRAIN_LIST_FILE%"

@REM REM Bardziej odporne sprawdzanie błędu
@REM if %errorlevel% neq 0 (
@REM     echo.
@REM     echo *** BLAD: Krok 1 (find_images_to_train.py) nie powiodl sie! (Errorlevel: %errorlevel%) ***
@REM     goto :error
@REM )
echo === Krok 1 zakonczony pomyslnie. ===

REM --- Krok 2: Pobierz obrazy, spłaszcz nazwy, podziel i stwórz mapowanie ---
echo.
echo === Krok 2: Przygotowanie struktury obrazow i mapowania nazw... ===
%PYTHON_EXE% "%SCRIPTS_DIR%\prepare_yolo_dataset.py" --container-name "%CONTAINER_NAME%" --input-file "%TRAIN_LIST_FILE%" --dataset-name "%DATASET_DIR%" --mapping-file "%MAPPING_FILENAME%"

@REM if %errorlevel% neq 0 (
@REM     echo.
@REM     echo *** BLAD: Krok 2 (prepare_yolo_dataset.py) nie powiodl sie! (Errorlevel: %errorlevel%) ***
@REM     goto :error
@REM )
echo === Krok 2 zakonczony pomyslnie. ===

REM --- Krok 3: Zorganizuj etykiety YOLO i stwórz pliki konfiguracyjne ---
echo.
echo === Krok 3: Organizacja etykiet i tworzenie plikow konfiguracyjnych YOLO... ===

REM Konstrukcja pełnej ścieżki do pliku mapowania
set MAPPING_FILE_FULLPATH=%DATASET_DIR%\%MAPPING_FILENAME%

%PYTHON_EXE% "%SCRIPTS_DIR%\organize_yolo_labels.py" --container-name "%CONTAINER_NAME%" --annotation-blobs %YOLO_ANNOTATION_BLOBS% --dataset-dir "%DATASET_DIR%" --mapping-file "%MAPPING_FILE_FULLPATH%" --class-names-file "%CLASS_NAMES_FILENAME%" --zip-base-structure "%YOLO_ZIP_BASE_STRUCTURE%"

@REM if %errorlevel% neq 0 (
@REM     echo.
@REM     echo *** BLAD: Krok 3 (organize_yolo_labels.py) nie powiodl sie! (Errorlevel: %errorlevel%) ***
@REM     goto :error
@REM )
echo === Krok 3 zakonczony pomyslnie. ===

echo.
echo === Wszystkie kroki zakonczone pomyslnie! ===
echo Dataset YOLO gotowy w folderze: %DATASET_DIR%
goto :eof

:error
echo.
echo Skrypt przerwany z powodu bledu.
exit /b 1

:eof
endlocal