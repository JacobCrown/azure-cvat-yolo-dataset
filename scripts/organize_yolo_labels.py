# -*- coding: utf-8 -*-
import os
from typing import Optional
import zipfile
import argparse
import sys
import shutil
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from tqdm import tqdm
import time
import json
import yaml  # Potrzebne do zapisu pliku YAML


# --- Funkcje pomocnicze (download, unzip, find_all_txt_files) - bez zmian ---
# (Wklej tutaj te funkcje z poprzedniej odpowiedzi)
def download_blob_sync(
    connect_str: str, container_name: str, blob_name: str, download_file_path: str
):
    if not connect_str:
        print("Błąd: Brak ciągu połączenia.", file=sys.stderr)
        return False
    try:
        print(f"  Łączenie z Azure dla {blob_name}...")
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        blob_client = blob_service_client.get_blob_client(
            container=container_name, blob=blob_name
        )
        if not blob_client.exists():
            print(f"  Błąd: Blob '{blob_name}' nie istnieje.", file=sys.stderr)
            return False
        print(f"  Pobieranie {blob_name} do {download_file_path}...")
        os.makedirs(os.path.dirname(download_file_path), exist_ok=True)
        with open(download_file_path, "wb") as download_file:
            download_stream = blob_client.download_blob()
            download_file.write(download_stream.readall())
        print(f"  Pobieranie {blob_name} zakończone.")
        return True
    except Exception as e:
        print(f"  Błąd pobierania {blob_name}: {e}", file=sys.stderr)
        return False


def unzip_file(zip_path: str, extract_to_path: str):
    try:
        os.makedirs(extract_to_path, exist_ok=True)
        print(f"  Rozpakowywanie {os.path.basename(zip_path)} do {extract_to_path}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to_path)
        print(f"  Rozpakowywanie {os.path.basename(zip_path)} zakończone.")
        return True
    except Exception as e:
        print(
            f"  Błąd rozpakowywania {os.path.basename(zip_path)}: {e}", file=sys.stderr
        )
        return False


def find_all_txt_files(directory: str) -> list[str]:
    txt_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            # Pomijamy train.txt i val.txt z archiwum, bo generujemy nowe
            if file.lower().endswith(".txt") and file.lower() not in [
                "train.txt",
                "val.txt",
            ]:
                txt_files.append(os.path.join(root, file))
    return txt_files


# --- Funkcja organize_labels - z poprzedniej odpowiedzi (z poprawnymi nazwami) ---
def organize_labels(
    source_txt_files: list[str],
    extract_base_path: str,
    dataset_base_dir: str,
    path_mapping: dict[str, str],
    image_ext: str,
    zip_base_structure: Optional[
        str
    ] = "obj_train_data",  # Typ Optional, bo może być None (jeśli nie podano)  # Nadal potrzebne do relatywnej ścieżki
) -> tuple[int, int, int]:
    """
    Kopiuje pliki .txt do odpowiednich folderów labels/train lub labels/valid,
    zapisując je pod nazwą odpowiadającą SPŁASZCZONEJ nazwie obrazu.
    Zwraca krotkę: (liczba_skopiowanych_train, liczba_skopiowanych_valid, liczba_pominietych)
    """
    train_labels_dir = os.path.join(dataset_base_dir, "labels", "train")
    valid_labels_dir = os.path.join(dataset_base_dir, "labels", "valid")
    train_images_dir = os.path.join(dataset_base_dir, "images", "train")
    valid_images_dir = os.path.join(dataset_base_dir, "images", "valid")

    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(valid_labels_dir, exist_ok=True)

    copied_train_count, copied_valid_count, skipped_count, map_key_not_found = (
        0,
        0,
        0,
        0,
    )

    for txt_path in tqdm(source_txt_files, desc="   Organizowanie .txt", unit="plik"):
        try:
            relative_txt_path_full = os.path.relpath(txt_path, extract_base_path)
        except ValueError:
            skipped_count += 1
            continue

        relative_txt_path = relative_txt_path_full.replace("\\", "/")
        if zip_base_structure:
            prefix_to_remove = zip_base_structure.replace("\\", "/") + "/"
            if relative_txt_path.startswith(prefix_to_remove):
                relative_txt_path = relative_txt_path[len(prefix_to_remove) :]

        key_base = os.path.splitext(relative_txt_path)[0]
        original_azure_key = f"{key_base}.{image_ext.lstrip('.')}"
        flattened_image_filename = path_mapping.get(original_azure_key)

        if not flattened_image_filename:
            map_key_not_found += 1
            skipped_count += 1
            continue

        expected_train_image_path = os.path.join(
            train_images_dir, flattened_image_filename
        )
        expected_valid_image_path = os.path.join(
            valid_images_dir, flattened_image_filename
        )

        target_labels_dir = None
        target_set = None
        if os.path.exists(expected_train_image_path):
            target_labels_dir = train_labels_dir
            target_set = "train"
        elif os.path.exists(expected_valid_image_path):
            target_labels_dir = valid_labels_dir
            target_set = "valid"
        else:
            skipped_count += 1
            continue

        flattened_label_base = os.path.splitext(flattened_image_filename)[0]
        flattened_label_filename = f"{flattened_label_base}.txt"
        destination_path = os.path.join(target_labels_dir, flattened_label_filename)

        try:
            shutil.copy2(txt_path, destination_path)
            if target_set == "train":
                copied_train_count += 1
            elif target_set == "valid":
                copied_valid_count += 1
        except Exception:
            skipped_count += 1

    # Podsumowanie dla tego archiwum (mniej gadatliwe)
    print(
        f"  Wynik org. etykiet: train={copied_train_count}, valid={copied_valid_count}, pominięte={skipped_count}, brak_mapy={map_key_not_found}"
    )
    return copied_train_count, copied_valid_count, skipped_count


# --- NOWA Funkcja do odczytu klas z obj.names ---
def read_class_names_from_obj_names(obj_names_path: str) -> Optional[list[str]]:
    """Odczytuje nazwy klas z pliku obj.names (lub podobnego)."""
    if not os.path.isfile(obj_names_path):
        print(
            f"  Ostrzeżenie: Plik '{os.path.basename(obj_names_path)}' nie został znaleziony. Nie można odczytać nazw klas.",
            file=sys.stderr,
        )
        return None
    try:
        print(
            f"  Odczytywanie nazw klas z pliku: {os.path.basename(obj_names_path)}..."
        )
        with open(obj_names_path, "r", encoding="utf-8") as f:
            # Usuwamy puste linie i białe znaki z początku/końca
            class_names = [line.strip() for line in f if line.strip()]
        if not class_names:
            print(
                f"  Ostrzeżenie: Plik '{os.path.basename(obj_names_path)}' jest pusty.",
                file=sys.stderr,
            )
            return None
        print(f"  Odczytano {len(class_names)} nazw klas: {', '.join(class_names)}")
        return class_names
    except Exception as e:
        print(
            f"  Błąd podczas odczytu pliku '{os.path.basename(obj_names_path)}': {e}",
            file=sys.stderr,
        )
        return None


# --- Funkcja create_yolo_config_files - bez zmian w logice, tylko przyjmuje class_names ---
def create_yolo_config_files(dataset_base_dir: str, class_names: list[str]):
    """Tworzy pliki train.txt, val.txt i dataset.yaml."""
    # (Skopiuj tę funkcję z poprzedniej odpowiedzi - działa poprawnie)
    print("\n--- Tworzenie plików konfiguracyjnych YOLO ---")
    train_img_dir = os.path.join(dataset_base_dir, "images", "train")
    valid_img_dir = os.path.join(dataset_base_dir, "images", "valid")
    train_txt_path = os.path.join(dataset_base_dir, "train.txt")
    valid_txt_path = os.path.join(dataset_base_dir, "val.txt")
    yaml_path = os.path.join(dataset_base_dir, "dataset.yaml")

    if not os.path.isdir(train_img_dir):
        print(f"Ostrzeżenie: Brak '{train_img_dir}'.", file=sys.stderr)
        return False
    if not os.path.isdir(valid_img_dir):
        print(f"Ostrzeżenie: Brak '{valid_img_dir}'.", file=sys.stderr)
        return False

    try:  # Generuj train.txt
        print(f"Generowanie {train_txt_path}...")
        train_image_files = [
            os.path.join("images", "train", f).replace("\\", "/")
            for f in os.listdir(train_img_dir)
            if os.path.isfile(os.path.join(train_img_dir, f))
        ]
        with open(train_txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(train_image_files) + "\n")
        print(f"Zapisano {len(train_image_files)} ścieżek do {train_txt_path}")
    except Exception as e:
        print(f"Błąd generowania {train_txt_path}: {e}", file=sys.stderr)
        return False

    try:  # Generuj val.txt
        print(f"Generowanie {valid_txt_path}...")
        valid_image_files = [
            os.path.join("images", "valid", f).replace("\\", "/")
            for f in os.listdir(valid_img_dir)
            if os.path.isfile(os.path.join(valid_img_dir, f))
        ]
        with open(valid_txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(valid_image_files) + "\n")
        print(f"Zapisano {len(valid_image_files)} ścieżek do {valid_txt_path}")
    except Exception as e:
        print(f"Błąd generowania {valid_txt_path}: {e}", file=sys.stderr)
        return False

    if not class_names:
        print("Błąd: Brak nazw klas dla dataset.yaml.", file=sys.stderr)
        return False

    yaml_data = {  # Generuj dataset.yaml
        "path": os.path.abspath(dataset_base_dir).replace("\\", "/"),
        "train": "train.txt",
        "val": "val.txt",
        "nc": len(class_names),
        "names": class_names,
    }
    try:
        print(f"Generowanie {yaml_path}...")
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(
                yaml_data,
                f,
                sort_keys=False,
                default_flow_style=None,
                allow_unicode=True,
            )
        print(f"Zapisano konfigurację do {yaml_path}")
    except Exception as e:
        print(f"Błąd generowania {yaml_path}: {e}", file=sys.stderr)
        return False

    print("--- Zakończono tworzenie plików konfiguracyjnych ---")
    return True


# --- Główna funkcja main() - zmodyfikowana logika pobierania klas ---
def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Pobiera ZIPy YOLO, organizuje etykiety (z poprawnymi nazwami), tworzy pliki konfiguracyjne (używając obj.names)."
    )
    parser.add_argument("--connect-str", help="Ciąg połączenia Azure (nadpisuje .env).")
    parser.add_argument(
        "--container-name", required=True, help="Nazwa kontenera Azure."
    )
    parser.add_argument(
        "--annotation-blobs",
        required=True,
        nargs="+",
        help="Nazwy plików ZIP z adnotacjami YOLO.",
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Ścieżka do folderu datasetu (z images/ i mapowaniem).",
    )
    parser.add_argument(
        "--mapping-file", required=True, help="Ścieżka do pliku JSON z mapowaniem nazw."
    )
    parser.add_argument(
        "--image-ext",
        default="jpeg",
        help="Rozszerzenie obrazów (bez kropki, domyślnie: 'jpeg').",
    )
    parser.add_argument(
        "--zip-base-structure",
        default="obj_train_data",
        help="Folder bazowy w ZIP do pominięcia (domyślnie: 'obj_train_data', '' jeśli brak).",
    )
    # NOWY argument do wskazania nazwy pliku z klasami
    parser.add_argument(
        "--class-names-file",
        default="obj.names",
        help="Nazwa pliku wewnątrz archiwum ZIP, z którego mają być odczytane nazwy klas (domyślnie: obj.names).",
    )

    args = parser.parse_args()

    # Wczytanie connection string i mapowania (bez zmian)
    connect_str = (
        args.connect_str
        if args.connect_str
        else os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    )
    if not connect_str:
        sys.exit("Błąd: Brak ciągu połączenia.")
    if not os.path.isdir(args.dataset_dir):
        sys.exit(f"Błąd: Folder '{args.dataset_dir}' nie istnieje.")
    if not os.path.isdir(
        os.path.join(args.dataset_dir, "images", "train")
    ) or not os.path.isdir(os.path.join(args.dataset_dir, "images", "valid")):
        sys.exit(
            f"Błąd: Brak 'images/train' lub 'images/valid' w '{args.dataset_dir}'."
        )
    try:
        with open(args.mapping_file, "r", encoding="utf-8") as f:
            full_path_map = json.load(f)
    except Exception as e:
        sys.exit(f"Błąd wczytywania mapowania '{args.mapping_file}': {e}")

    # Zmienne do śledzenia
    total_copied_train, total_copied_valid, total_skipped = 0, 0, 0
    total_processed_zip, total_errors_zip = 0, 0
    class_names = None  # Lista klas odczytana z pliku
    class_names_source_file = None  # Zapamiętamy, skąd wzięliśmy klasy

    # Folder tymczasowy
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    temp_base_dir = f"temp_labels_objnames_{timestamp}"  # Zmieniona nazwa
    download_dir = os.path.join(temp_base_dir, "downloads")
    extract_base_dir = os.path.join(temp_base_dir, "extracted")
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(extract_base_dir, exist_ok=True)
    print(f"Używanie folderu tymczasowego: {temp_base_dir}")

    # --- Pętla przetwarzania ZIPów ---
    try:
        for blob_name in args.annotation_blobs:
            print(f"\n--- Przetwarzanie archiwum: {blob_name} ---")
            safe_blob_name = os.path.basename(blob_name)
            download_path = os.path.join(download_dir, safe_blob_name)
            extract_dir_name = safe_blob_name.replace(".zip", "")
            extract_path = os.path.join(
                extract_base_dir, extract_dir_name
            )  # Unikalny folder

            if not download_blob_sync(
                connect_str, args.container_name, blob_name, download_path
            ):
                total_errors_zip += 1
                continue
            if not unzip_file(download_path, extract_path):
                total_errors_zip += 1
                continue

            # --- Odczyt nazw klas (tylko raz, z pierwszego udanego pliku obj.names) ---
            if class_names is None:
                # Szukamy pliku np. obj.names w głównym folderze rozpakowanego ZIPa
                obj_names_path_in_zip = os.path.join(
                    extract_path, args.class_names_file
                )
                class_names = read_class_names_from_obj_names(obj_names_path_in_zip)
                if class_names:
                    class_names_source_file = f"{blob_name}/{args.class_names_file}"
                # else: # Komunikat o braku pliku jest już w read_class_names_from_obj_names

            # Znajdź i zorganizuj pliki .txt (etykiety)
            source_txt_files = find_all_txt_files(extract_path)
            if not source_txt_files:
                print(
                    f"  Ostrzeżenie: Nie znaleziono plików etykiet .txt w {blob_name}."
                )
            else:
                print(
                    f"  Znaleziono {len(source_txt_files)} plików .txt do organizacji."
                )
                copied_train, copied_valid, skipped = organize_labels(
                    source_txt_files,
                    extract_path,
                    args.dataset_dir,
                    full_path_map,
                    args.image_ext,
                    args.zip_base_structure if args.zip_base_structure else None,
                )
                total_copied_train += copied_train
                total_copied_valid += copied_valid
                total_skipped += skipped

            total_processed_zip += 1
            print(f"--- Zakończono przetwarzanie archiwum: {blob_name} ---")

        # --- Koniec pętli - Podsumowanie i tworzenie plików konfiguracyjnych ---
        print(f"\n--- Zakończono przetwarzanie wszystkich archiwów ---")
        print(
            f"Przetworzono {total_processed_zip} archiwów ZIP (błędy: {total_errors_zip})."
        )
        print(f"\nPodsumowanie organizacji etykiet:")
        print(f"  Skopiowano do labels/train: {total_copied_train}")
        print(f"  Skopiowano do labels/valid: {total_copied_valid}")
        print(f"  Pominięto etykiet: {total_skipped}")

        if class_names:
            print(f"\nNazwy klas zostały odczytane z pliku: {class_names_source_file}")
            config_success = create_yolo_config_files(args.dataset_dir, class_names)
            if config_success:
                print(
                    f"\nDataset w '{args.dataset_dir}' jest gotowy do użycia w treningu YOLO."
                )
            else:
                print(
                    f"\nUWAGA: Błędy podczas tworzenia plików konfiguracyjnych YOLO.",
                    file=sys.stderr,
                )
        else:
            print(
                f"\nBŁĄD KRYTYCZNY: Nie udało się odczytać nazw klas z pliku '{args.class_names_file}' z żadnego archiwum.",
                file=sys.stderr,
            )
            print(
                "Nie można wygenerować pliku dataset.yaml. Sprawdź zawartość plików ZIP.",
                file=sys.stderr,
            )

    finally:
        # Sprzątanie
        print("\nSprzątanie plików tymczasowych...")
        try:
            if os.path.exists(temp_base_dir):
                shutil.rmtree(temp_base_dir)
                print(f"Usunięto folder: {temp_base_dir}")
        except Exception as e:
            print(f"Błąd sprzątania '{temp_base_dir}': {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
