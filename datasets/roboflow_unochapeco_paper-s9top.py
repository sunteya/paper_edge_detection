import argparse
from roboflow_utils import download_roboflow_dataset

def main():
    parser = argparse.ArgumentParser(description='Download paper dataset from Roboflow')
    parser.add_argument('--overwrite', type=bool, help='Overwrite existing dataset')
    args = parser.parse_args()

    download_roboflow_dataset("unochapeco", "paper-s9top", "1", args.overwrite)

if __name__ == "__main__":
    main