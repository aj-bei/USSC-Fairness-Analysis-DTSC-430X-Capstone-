
import gdown
import os

def download_files_from_fld(folder_id: str, destination: str) -> None:
    """
    Downloads all files from a Google Drive folder given its folder ID. 
    Only downloads if files are not already present, which is checked 
    by comparing the files in the destination directory with those in 
    the Google Drive folder.

    Args:
        folder_id (str): The Google Drive folder ID.
        destination (str): The local directory where files will be saved.
    """

    url = f"https://drive.google.com/drive/folders/{folder_id}"
    fs_in_drive = set([f.path for f in gdown.download_folder(url, use_cookies=False, skip_download=True, quiet=True)])
    
    if set(os.listdir(destination)) > fs_in_drive:  # destination has all files plus extras
        print("All files already downloaded.")
        return
    print(f"Downloading files: {fs_in_drive - set(os.listdir(destination))}")
    gdown.download_folder(url, output=destination, quiet=False, use_cookies=False)


if __name__ == "__main__":
    # only runs if running this file directly
    folder_id = "1P2FRAkPrqL2nn2MNMyd4ilWbXNS_kkKD"
    destination = os.path.join(os.path.dirname(__file__), "../data")
    download_files_from_fld(folder_id, destination)