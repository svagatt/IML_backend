from EEGTools.Recorders.LiveAmpRecorder.liveamp_recorder import LiveAmpRecorder as Recorder
from read_save_data_files import get_path

import asyncio

# initialize the recorder and connect it
rec = Recorder()
rec.connect()


async def start_recording():
    rec.start_recording()
    await asyncio.sleep(3)


def set_event_with_offset(timer, sample_rate):
    offset = sample_rate * timer
    rec.set_event(99, -offset)


def get_latest_data_from_buffer():
    rec.refresh()
    data = rec.get_new_data()
    return data


def stop_recording(subject_id):
    rec.stop_recording()
    print('Recording has been stopped!')
    rec.disconnect()
    rec.save(file_prefix=f"subject_{subject_id}_raw", path=get_path('online_module_data'), description='Online Module Data Recording')
    rec.refresh()
    rec.clear()
