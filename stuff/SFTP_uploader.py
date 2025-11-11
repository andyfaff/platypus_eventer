import subprocess
import getpass
import time


def main():
    while True:
        result=subprocess.call("./merge.sh", shell=True)
        time.sleep(180.)


if __name__ == "__main__":
    main()
