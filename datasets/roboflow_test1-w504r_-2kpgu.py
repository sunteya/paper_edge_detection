import argparse
from dotenv import load_dotenv
from roboflow_utils import download_roboflow_dataset

def main():
    parser = argparse.ArgumentParser(description='Download paper dataset from Roboflow')
    parser.add_argument('--overwrite', type=bool, help='Overwrite existing dataset')
    args = parser.parse_args()
    
    load_dotenv()
    download_roboflow_dataset("test1-w504r", "-2kpgu", "1", args.overwrite)

if __name__ == "__main__":
    main()