from SoccerNet.Downloader import SoccerNetDownloader

download_path = "data/raw/SoccerNet"

downloader = SoccerNetDownloader(
    LocalDirectory=download_path
)

downloader.downloadDataTask(
    task="tracking-2023",
    split=["train"]
)

print("SoccerNet tracking training data downloaded.")