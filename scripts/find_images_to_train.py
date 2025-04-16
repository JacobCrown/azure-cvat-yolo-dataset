import os
from typing import Optional
import zipfile
import xml.etree.ElementTree as ET
import argparse
import sys
import shutil
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
import time  # Dodane do tworzenia unikalnych nazw folderów

# --- Funkcje pomocnicze (download, unzip, find_xml) - pozostają prawie bez zmian ---


def download_blob_sync(
    connect_str: str, container_name: str, blob_name: str, download_file_path: str
):
    """Pobiera synchronicznie plik blob z Azure Storage."""
    if not connect_str:
        print("Błąd krytyczny: Brak ciągu połączenia Azure Storage.", file=sys.stderr)
        return False
    try:
        print(f"  Łączenie z Azure Storage dla bloba: {blob_name}...")
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        blob_client = blob_service_client.get_blob_client(
            container=container_name, blob=blob_name
        )

        # Sprawdź czy blob istnieje przed próbą pobrania
        if not blob_client.exists():
            print(
                f"  Błąd: Blob '{blob_name}' nie istnieje w kontenerze '{container_name}'.",
                file=sys.stderr,
            )
            return False

        print(f"  Pobieranie pliku blob '{blob_name}' do '{download_file_path}'...")
        os.makedirs(
            os.path.dirname(download_file_path), exist_ok=True
        )  # Utwórz folder downloads, jeśli trzeba
        with open(download_file_path, "wb") as download_file:
            download_stream = blob_client.download_blob()
            download_file.write(download_stream.readall())
        print(f"  Pobieranie '{blob_name}' zakończone pomyślnie.")
        return True
    except ValueError as e:
        print(
            f"  Błąd: Problem z ciągiem połączenia lub nazwą kontenera/bloba '{blob_name}': {e}",
            file=sys.stderr,
        )
        return False
    except Exception as e:
        print(
            f"  Błąd podczas pobierania pliku blob '{blob_name}': {e}", file=sys.stderr
        )
        return False


def unzip_file(zip_path: str, extract_to_path: str):
    """Rozpakowuje plik ZIP do wskazanego folderu."""
    try:
        os.makedirs(extract_to_path, exist_ok=True)
        print(
            f"  Rozpakowywanie pliku '{os.path.basename(zip_path)}' do '{extract_to_path}'..."
        )
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to_path)
        print(f"  Rozpakowywanie '{os.path.basename(zip_path)}' zakończone.")
        return True
    except zipfile.BadZipFile:
        print(
            f"  Błąd: Plik '{os.path.basename(zip_path)}' nie jest poprawnym plikiem ZIP.",
            file=sys.stderr,
        )
        return False
    except FileNotFoundError:
        print(
            f"  Błąd: Plik ZIP '{os.path.basename(zip_path)}' nie został znaleziony (prawdopodobnie problem z pobieraniem).",
            file=sys.stderr,
        )
        return False
    except Exception as e:
        print(
            f"  Błąd podczas rozpakowywania pliku '{os.path.basename(zip_path)}': {e}",
            file=sys.stderr,
        )
        return False


def find_xml_file(directory: str) -> Optional[str]:
    """Znajduje pierwszy plik .xml w podanym folderze i jego podfolderach."""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".xml"):
                return os.path.join(root, file)
    return None


# --- Zmodyfikowana funkcja przetwarzania XML ---
def extract_training_images_from_xml(xml_file_path: str) -> Optional[list[str]]:
    """
    Przetwarza plik XML, znajduje obrazy do treningu (mające BBoxy LUB tag 'brak reklam')
    i ZWRACA listę ich nazw. Zwraca None w przypadku błędu.
    """
    images_to_keep = []
    try:
        print(f"  Przetwarzanie pliku XML: {xml_file_path}...")
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        image_elements = root.findall(".//image")
        # print(f"  Znaleziono {len(image_elements)} elementów <image> w pliku XML.") # Mniej gadatliwy output

        for i, image_elem in enumerate(image_elements):
            image_name = image_elem.get("name")
            if not image_name:
                print(
                    f"  Ostrzeżenie: Element <image> nr {i+1} w pliku {os.path.basename(xml_file_path)} nie ma atrybutu 'name'. Pomijanie.",
                    file=sys.stderr,
                )
                continue

            has_bboxes = image_elem.find("box") is not None
            has_brak_reklam_tag = any(
                tag_elem.get("label") == "brak reklam"
                for tag_elem in image_elem.findall("tag")
            )

            if has_bboxes or has_brak_reklam_tag:
                images_to_keep.append(image_name)
                # Opcjonalnie: Można dodać cichsze logowanie znalezionych plików
                # reason = []
                # if has_bboxes: reason.append("bbox")
                # if has_brak_reklam_tag: reason.append("brak_reklam")
                # print(f"    + {image_name} ({','.join(reason)})")

        print(
            f"  Zakończono przetwarzanie XML: {os.path.basename(xml_file_path)}. Znaleziono {len(images_to_keep)} pasujących obrazów."
        )
        return images_to_keep

    except ET.ParseError as e:
        print(
            f"  Błąd parsowania pliku XML '{os.path.basename(xml_file_path)}': {e}",
            file=sys.stderr,
        )
        return None
    except FileNotFoundError:
        print(
            f"  Błąd: Plik XML '{os.path.basename(xml_file_path)}' nie został znaleziony (ścieżka: {xml_file_path}).",
            file=sys.stderr,
        )
        return None
    except Exception as e:
        print(
            f"  Nieoczekiwany błąd podczas przetwarzania XML '{os.path.basename(xml_file_path)}': {e}",
            file=sys.stderr,
        )
        return None


# --- Główna funkcja ---
def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Pobiera WIELE plików ZIP z Azure, rozpakowuje je i znajduje obrazy do uczenia (mające BBoxy lub tag 'brak reklam'), zapisując wyniki do JEDNEGO pliku. Używa .env dla danych Azure."
    )
    parser.add_argument(
        "--connect-str",
        help="Ciąg połączenia Azure Storage (nadpisuje wartość z .env).",
    )
    parser.add_argument(
        "--container-name", required=True, help="Nazwa kontenera w Azure Blob Storage."
    )
    # Zmieniony argument --blob-name na --blob-names z nargs='+'
    parser.add_argument(
        "--blob-names",
        required=True,
        nargs="+",
        help="Jedna lub więcej nazw plików ZIP (blobów) w kontenerze, oddzielone spacjami.",
    )
    parser.add_argument(
        "--output-file",
        default="to_train_combined.txt",
        help="Nazwa pliku wyjściowego z połączoną listą obrazów do uczenia (domyślnie: to_train_combined.txt).",
    )

    args = parser.parse_args()

    connect_str = (
        args.connect_str
        if args.connect_str
        else os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    )

    if not connect_str:
        print(
            "Błąd: Ciąg połączenia Azure Storage nie został podany ani jako argument --connect-str, ani w pliku .env jako AZURE_STORAGE_CONNECTION_STRING.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Użyjemy zbioru (set) do przechowywania nazw obrazów, aby automatycznie obsłużyć duplikaty
    all_images_to_keep_set = set()
    total_processed_xml = 0
    total_errors = 0

    # Utwórz główny folder tymczasowy z unikalnym znacznikiem czasu, aby uniknąć konfliktów
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    temp_base_dir = f"temp_annotations_multi_{timestamp}"
    download_dir = os.path.join(temp_base_dir, "downloads")
    extract_base_dir = os.path.join(temp_base_dir, "extracted")

    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(extract_base_dir, exist_ok=True)
    print(f"Używanie folderu tymczasowego: {temp_base_dir}")

    try:
        # Pętla przetwarzająca każdy podany plik blob
        for blob_name in args.blob_names:
            print(f"\n--- Rozpoczynanie przetwarzania bloba: {blob_name} ---")
            error_occurred_for_blob = False

            # Unikalna nazwa dla pobranego pliku i folderu ekstrakcji
            safe_blob_name = os.path.basename(blob_name)
            download_path = os.path.join(download_dir, safe_blob_name)
            # Tworzymy unikalny folder dla każdego ZIPa, np. na podstawie nazwy pliku
            extract_dir_name = safe_blob_name.replace(".zip", "")
            extract_path = os.path.join(extract_base_dir, extract_dir_name)

            # Krok 1: Pobierz plik z Azure
            if not download_blob_sync(
                connect_str, args.container_name, blob_name, download_path
            ):
                print(f"### Błąd pobierania {blob_name}. Pomijanie tego bloba. ###")
                total_errors += 1
                continue  # Przejdź do następnego bloba

            # Krok 2: Rozpakuj plik ZIP
            if not unzip_file(download_path, extract_path):
                print(f"### Błąd rozpakowywania {blob_name}. Pomijanie tego bloba. ###")
                total_errors += 1
                continue  # Przejdź do następnego bloba

            # Krok 3: Znajdź plik XML w folderze ekstrakcji *tego* ZIPa
            xml_file = find_xml_file(extract_path)
            if not xml_file:
                print(
                    f"### Ostrzeżenie: Nie znaleziono pliku XML w rozpakowanym archiwum dla {blob_name} w '{extract_path}'. Pomijanie tego bloba. ###"
                )
                total_errors += 1
                continue  # Przejdź do następnego bloba
            print(
                f"  Znaleziono plik XML: {os.path.relpath(xml_file, temp_base_dir)}"
            )  # Krótsza ścieżka dla logów

            # Krok 4: Przetwórz plik XML i dodaj wyniki do zbioru
            images_from_this_xml = extract_training_images_from_xml(xml_file)

            if images_from_this_xml is not None:
                before_update_count = len(all_images_to_keep_set)
                all_images_to_keep_set.update(
                    images_from_this_xml
                )  # .update() dodaje elementy ze zwracanej listy do zbioru
                added_count = len(all_images_to_keep_set) - before_update_count
                print(
                    f"  Dodano {added_count} unikalnych nazw obrazów z {os.path.basename(xml_file)}. Łącznie unikalnych: {len(all_images_to_keep_set)}"
                )
                total_processed_xml += 1
            else:
                print(
                    f"### Błąd przetwarzania XML dla {blob_name}. Wyniki z tego pliku nie zostaną dodane. ###"
                )
                total_errors += 1
                continue  # Przejdź do następnego bloba

            print(f"--- Zakończono przetwarzanie bloba: {blob_name} ---")

        # Po zakończeniu pętli - zapisz zgromadzone wyniki
        print(f"\n--- Zakończono przetwarzanie wszystkich blobów ---")
        print(f"Przetworzono poprawnie {total_processed_xml} plików XML.")
        if total_errors > 0:
            print(
                f"Wystąpiło {total_errors} błędów podczas przetwarzania niektórych blobów (szczegóły powyżej)."
            )

        print(
            f"Łącznie znaleziono {len(all_images_to_keep_set)} unikalnych obrazów do treningu."
        )
        print(f"Zapisywanie połączonej listy obrazów do pliku: {args.output_file}...")

        # Konwertuj zbiór z powrotem na listę (opcjonalnie sortuj dla spójności)
        final_image_list = sorted(list(all_images_to_keep_set))

        with open(args.output_file, "w", encoding="utf-8") as f:
            for name in final_image_list:
                f.write(name + "\n")

        print(
            f"Zapisano {len(final_image_list)} nazw obrazów do pliku '{args.output_file}'."
        )
        print(f"\nOperacja zakończona pomyślnie.")

    finally:
        # Krok 5: Sprzątanie - usuń cały główny folder tymczasowy
        print("\nSprzątanie plików tymczasowych...")
        try:
            if os.path.exists(temp_base_dir):
                shutil.rmtree(temp_base_dir)
                print(f"Usunięto folder tymczasowy: {temp_base_dir}")
            else:
                print(f"Folder tymczasowy '{temp_base_dir}' nie istnieje.")
        except OSError as e:
            print(
                f"Ostrzeżenie: Nie można usunąć folderu tymczasowego '{temp_base_dir}': {e}",
                file=sys.stderr,
            )
        except Exception as e:
            print(f"Nieoczekiwany błąd podczas sprzątania: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
