import requests
import os
import zipfile

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == "__main__":
    file_id = '1gEcyb4HUDzIvu7lQWTOyDC1X00YzCxFx'
    working_directory = os.getcwd()

    if not os.path.exists(os.path.join(working_directory, "pretrained_model")):
        os.makedirs(os.path.join(working_directory, "pretrained_model"))

    destination = os.path.join(working_directory, "pretrained_model", "imagenet21k+imagenet2012_ViT-B_16.pth")
    download_file_from_google_drive(file_id, destination)