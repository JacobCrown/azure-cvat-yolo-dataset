import os
import random
import math
import argparse
import sys
from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from tqdm import tqdm
import re  # Do zamiany wielu myślników
import json  # Do zapisania mapowania


def read_image_list(file_path: str) -> list[str]:
    """Wczytuje listę ścieżek obrazów z pliku tekstowego."""
    # (Bez zmian w stosunku do poprzedniej wersji)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        print(f"Odczytano {len(lines)} ścieżek obrazów z pliku '{file_path}'.")
        return lines
    except FileNotFoundError:
        print(
            f"Błąd: Plik wejściowy '{file_path}' nie został znaleziony.",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(f"Błąd podczas odczytu pliku '{file_path}': {e}", file=sys.stderr)
        sys.exit(1)


def split_data(
    image_paths: list[str], valid_split_ratio: float
) -> tuple[list[str], list[str]]:
    """Dzieli listę obrazów na zbiory treningowy i walidacyjny."""
    # (Bez zmian w stosunku do poprzedniej wersji)
    if not 0.0 <= valid_split_ratio <= 1.0:
        print(
            "Błąd: Współczynnik podziału walidacyjnego musi być pomiędzy 0.0 a 1.0.",
            file=sys.stderr,
        )
        sys.exit(1)
    random.shuffle(image_paths)
    num_total = len(image_paths)
    num_valid = math.ceil(num_total * valid_split_ratio)
    num_train = num_total - num_valid
    valid_files = image_paths[:num_valid]
    train_files = image_paths[num_valid:]
    print(
        f"Podział danych: {num_train} obrazów treningowych, {num_valid} obrazów walidacyjnych ({valid_split_ratio*100:.1f}%)."
    )
    return train_files, valid_files


def create_yolo_dirs(base_dir: str):
    """Tworzy strukturę folderów dla YOLO (images/train, images/valid)."""
    # (Bez zmian w stosunku do poprzedniej wersji)
    train_img_dir = os.path.join(base_dir, "images", "train")
    valid_img_dir = os.path.join(base_dir, "images", "valid")
    try:
        os.makedirs(train_img_dir, exist_ok=True)
        os.makedirs(valid_img_dir, exist_ok=True)
        print(f"Utworzono strukturę folderów w: {base_dir}")
        return train_img_dir, valid_img_dir
    except OSError as e:
        print(f"Błąd podczas tworzenia folderów w '{base_dir}': {e}", file=sys.stderr)
        sys.exit(1)


def flatten_azure_path(azure_path: str) -> str:
    """Konwertuje ścieżkę Azure blob na spłaszczoną, bezpieczną nazwę pliku."""
    flat_path = azure_path.lower()
    # Zamień slashe (forward i backward) na myślniki
    flat_path = flat_path.replace("/", "-").replace("\\", "-")
    # Zamień potencjalne wielokrotne myślniki na pojedynczy
    flat_path = re.sub(r"-+", "-", flat_path)
    # Usuń myślniki z początku/końca, jeśli powstały
    flat_path = flat_path.strip("-")
    return flat_path


def download_images(
    connect_str: str,
    container_name: str,
    file_list: list[str],
    destination_dir: str,
    set_name: str,
) -> tuple[dict[str, str], bool]:
    """
    Pobiera listę obrazów z Azure do wskazanego folderu lokalnego,
    ZMIENIAJĄC nazwy plików na spłaszczone ścieżki Azure.
    Zwraca mapowanie {oryginalna_sciezka_azure: nowa_spłaszczona_nazwa_pliku} oraz status powodzenia.
    """
    if not connect_str:
        print("Błąd krytyczny: Brak ciągu połączenia Azure Storage.", file=sys.stderr)
        return {}, False  # Zwracamy pusty słownik i False

    path_mapping = {}  # Słownik do przechowywania mapowania
    success_count = 0
    error_count = 0
    not_found_count = 0

    try:
        print(
            f"\nRozpoczynanie pobierania i zmiany nazw {len(file_list)} obrazów dla zbioru '{set_name}' do '{destination_dir}'..."
        )
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        container_client = blob_service_client.get_container_client(container_name)

        for azure_path in tqdm(file_list, desc=f"Pobieranie ({set_name})", unit="plik"):
            # Generuj NOWĄ, spłaszczoną nazwę pliku
            new_flat_filename = flatten_azure_path(azure_path)
            local_path = os.path.join(destination_dir, new_flat_filename)

            try:
                blob_client = container_client.get_blob_client(blob=azure_path)

                # Sprawdź czy plik docelowy (z nową nazwą) już istnieje
                if os.path.exists(local_path):
                    # print(f"  Plik '{new_flat_filename}' już istnieje lokalnie. Pomijanie.")
                    # Mimo pominięcia pobierania, nadal dodajemy do mapowania, bo plik istnieje
                    path_mapping[azure_path] = new_flat_filename
                    success_count += 1
                    continue

                with open(local_path, "wb") as download_file:
                    download_stream = blob_client.download_blob()
                    download_file.write(download_stream.readall())

                # Pobieranie udane, dodaj do mapowania
                path_mapping[azure_path] = new_flat_filename
                success_count += 1

            except ResourceNotFoundError:
                print(
                    f"\n  Ostrzeżenie: Blob '{azure_path}' nie został znaleziony w kontenerze '{container_name}'. Pomijanie.",
                    file=sys.stderr,
                )
                not_found_count += 1
                if os.path.exists(local_path) and os.path.getsize(local_path) == 0:
                    os.remove(local_path)
            except Exception as e:
                print(
                    f"\n  Błąd podczas pobierania bloba '{azure_path}': {e}",
                    file=sys.stderr,
                )
                error_count += 1
                if os.path.exists(local_path):
                    try:
                        os.remove(local_path)
                    except OSError:
                        pass

        print(f"\nZakończono pobieranie dla zbioru '{set_name}'.")
        print(f"  Pobranych/istniejących pomyślnie: {success_count}")
        if not_found_count > 0:
            print(f"  Nie znaleziono w Azure: {not_found_count}")
        if error_count > 0:
            print(f"  Błędy pobierania: {error_count}")

        # Zwróć mapowanie i ogólny status powodzenia (true jeśli bez błędów i braków)
        return path_mapping, error_count == 0 and not_found_count == 0

    except Exception as e:
        print(
            f"\nKrytyczny błąd podczas procesu pobierania dla zbioru '{set_name}': {e}",
            file=sys.stderr,
        )
        return {}, False  # Zwróć pusty słownik i False


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Przygotowuje strukturę datasetu YOLOv9, pobierając obrazy z Azure, ZMIENIAJĄC ich nazwy na spłaszczone ścieżki, dzieląc na train/valid i zapisując mapowanie nazw."
    )
    # Argumenty jak poprzednio
    parser.add_argument(
        "--input-file",
        required=True,
        help="Plik tekstowy z listą oryginalnych ścieżek obrazów Azure (np. to_train_combined.txt).",
    )
    parser.add_argument(
        "--connect-str",
        help="Ciąg połączenia Azure Storage (nadpisuje wartość z .env).",
    )
    parser.add_argument(
        "--container-name", required=True, help="Nazwa kontenera w Azure Blob Storage."
    )
    parser.add_argument(
        "--dataset-name",
        default="yolo_dataset",
        help="Nazwa folderu głównego dla tworzonego datasetu (domyślnie: yolo_dataset).",
    )
    parser.add_argument(
        "--valid-split",
        type=float,
        default=0.1,
        help="Procent danych do przeznaczenia na zbiór walidacyjny (od 0.0 do 1.0, domyślnie: 0.1 czyli 10%).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Ziarno losowości dla podziału train/valid (dla powtarzalności).",
    )
    # Dodatkowy argument na nazwę pliku mapowania
    parser.add_argument(
        "--mapping-file",
        default="azure_to_local_map.json",
        help="Nazwa pliku JSON do zapisania mapowania oryginalnych ścieżek Azure na nowe lokalne nazwy (domyślnie: azure_to_local_map.json w folderze datasetu).",
    )

    args = parser.parse_args()

    random.seed(args.random_seed)
    print(f"Użyto ziarna losowości: {args.random_seed}")

    connect_str = (
        args.connect_str
        if args.connect_str
        else os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    )
    if not connect_str:
        print("Błąd: Ciąg połączenia Azure Storage nie został podany.", file=sys.stderr)
        sys.exit(1)

    all_image_paths = read_image_list(args.input_file)
    if not all_image_paths:
        print("Lista obrazów jest pusta. Przerywanie.", file=sys.stderr)
        sys.exit(1)

    train_files, valid_files = split_data(all_image_paths, args.valid_split)

    train_img_dir, valid_img_dir = create_yolo_dirs(args.dataset_name)

    # Pobieranie i zbieranie mapowań
    print("\nPobieranie obrazów treningowych (ze zmianą nazw)...")
    train_map, train_success = download_images(
        connect_str, args.container_name, train_files, train_img_dir, "train"
    )

    print("\nPobieranie obrazów walidacyjnych (ze zmianą nazw)...")
    valid_map, valid_success = download_images(
        connect_str, args.container_name, valid_files, valid_img_dir, "valid"
    )

    # Połącz mapowania
    full_path_map = {**train_map, **valid_map}  # Łączenie słowników

    # Zapisz mapowanie do pliku JSON w folderze datasetu
    mapping_filepath = os.path.join(args.dataset_name, args.mapping_file)
    try:
        print(
            f"\nZapisywanie mapowania {len(full_path_map)} ścieżek do pliku: {mapping_filepath}"
        )
        with open(mapping_filepath, "w", encoding="utf-8") as f:
            json.dump(full_path_map, f, indent=4, ensure_ascii=False)
        print("Mapowanie zapisane pomyślnie.")
    except Exception as e:
        print(
            f"\nBłąd podczas zapisywania pliku mapowania '{mapping_filepath}': {e}",
            file=sys.stderr,
        )
        print(
            "OSTRZEŻENIE: Plik mapowania jest niezbędny do poprawnego dopasowania etykiet w następnym kroku!"
        )

    # Podsumowanie
    print("\n--- Zakończono przygotowanie datasetu ze spłaszczonymi nazwami ---")
    if train_success and valid_success:
        print(
            f"Obrazy zostały pomyślnie pobrane (lub istniały) i umieszczone w folderze '{args.dataset_name}' z nowymi, unikalnymi nazwami."
        )
    else:
        print(
            f"UWAGA: Wystąpiły problemy podczas pobierania niektórych obrazów. Dataset w folderze '{args.dataset_name}' może być niekompletny."
        )

    print(f"\nStruktura folderów dla obrazów:")
    print(f"- Trening: {os.path.abspath(train_img_dir)}")
    print(f"- Walidacja: {os.path.abspath(valid_img_dir)}")
    print(
        f"\nWAŻNE: Nazwy plików obrazów zostały zmienione na spłaszczone ścieżki Azure."
    )
    print(f"Mapowanie oryginalnych ścieżek Azure na nowe nazwy lokalne zapisano w:")
    print(f"  {os.path.abspath(mapping_filepath)}")
    print(
        "Ten plik mapowania będzie potrzebny w następnym kroku do organizacji etykiet."
    )


if __name__ == "__main__":
    main()
