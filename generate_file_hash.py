import hashlib
import os


def sha256sum(filename):
    h = hashlib.sha256()
    b = bytearray(128*1024)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()


def get_hash_for_preprocessed_data(sub_id, channel_num, is_online, ctr):
    if is_online:
        file_path = f'{os.getcwd()}/preprocessed_data/sub_{sub_id}_preprocessed_{channel_num}_{ctr}raw.fif'
    else:
        file_path = f'{os.getcwd()}/preprocessed_data/sub_{sub_id}_preprocessed_{channel_num}_raw.fif'
    return sha256sum(file_path)
