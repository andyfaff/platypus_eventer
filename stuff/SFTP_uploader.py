import os
import time
import paramiko
from pathlib import Path
import getpass
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, DirCreatedEvent



# === CONFIGURATION ===
LOCAL_BASE_DIR = os.getcwd()
REMOTE_BASE_DIR = "/home/quokka/event/SEF"
KNOWN_DIRS_FILE = "uploaded_dirs.txt"

SFTP_HOST = "ics1-quokka.nbi.ansto.gov.au"
SFTP_PORT = 22
SFTP_USER = getpass.getpass("Username:")
SFTP_PASS = getpass.getpass("Password:")


def connect_sftp():
    """Establish and return an SFTP connection."""
    transport = paramiko.Transport((SFTP_HOST, SFTP_PORT))
    transport.connect(username=SFTP_USER, password=SFTP_PASS)
    return paramiko.SFTPClient.from_transport(transport)


def upload_directory(sftp, local_dir, remote_dir):
    """Recursively upload a directory and its files."""
    for root, dirs, files in os.walk(local_dir):
        rel_path = os.path.relpath(root, local_dir)
        remote_path = os.path.join(remote_dir, rel_path).replace("\\", "/")
        # remove /.
        remote_path = remote_path.rstrip("/.")
        # Ensure remote directory exists
        try:
            sftp.stat(remote_path)
        except FileNotFoundError:
            sftp.mkdir(remote_path)

        for file in files:
            local_file = os.path.join(root, file)
            remote_file = os.path.join(remote_path, file).replace("\\", "/")
            print(f"Uploading {local_file} → {remote_file}")
            sftp.put(local_file, remote_file)


class MyEventHandler(FileSystemEventHandler):
    self.last_time = time.time()

    def on_created(self, event):
        self.upload(event)

    def on_modified(self, event):
        self.upload(event)

    def upload(self, event):
        if time.time() > self.last_time + 60:
            self.last_time = time.time()
            pth = Path(event.src_path)
            top_dir = pth.parts[0]
            try:
                sftp = connect_sftp()
                for local_dir in new_dirs:
                    rel_path = os.path.relpath(local_dir, LOCAL_BASE_DIR)
                    remote_dir = os.path.join(REMOTE_BASE_DIR, rel_path).replace("\\", "/")
                    upload_directory(sftp, local_dir, remote_dir)
                    known_dirs.add(local_dir)
                sftp.close()
            except Exception as e:
                print(f"[ERROR] Upload failed: {e}")



def main():
    event_handler = MyEventHandler()
    observer = Observer()
    observer.schedule(event_handler, ".", recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    finally:
        observer.stop()
        observer.join()
        
    # known_dirs = load_known_dirs()
    # print(f"Loaded {len(known_dirs)} known directories.")

    # while True:
    #     current_dirs = {
    #         str(p) for p in Path(LOCAL_BASE_DIR).rglob("DAQ*") if p.is_dir()
    #     }
    #     new_dirs = current_dirs - known_dirs
    #     current_datetime = datetime.now()
    #     iso_formatted_time = current_datetime.isoformat()
    #     print(iso_formatted_time)
    #     if new_dirs:
    #         print(f"Found new directories.: {new_dirs}")

    #         try:
    #             sftp = connect_sftp()
    #             for local_dir in new_dirs:
    #                 rel_path = os.path.relpath(local_dir, LOCAL_BASE_DIR)
    #                 remote_dir = os.path.join(REMOTE_BASE_DIR, rel_path).replace("\\", "/")
    #                 upload_directory(sftp, local_dir, remote_dir)
    #                 known_dirs.add(local_dir)
    #             sftp.close()
    #             save_known_dirs(known_dirs)
    #         except Exception as e:
    #             print(f"[ERROR] Upload failed: {e}")
    #     else:
    #         print("No new directories found.")

    #     time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    main()
