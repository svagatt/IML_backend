from EEGTools.Recorders.LiveAmpRecorder.liveamp_recorder import LiveAmpRecorder as Recorder
from read_save_data_files import get_path
from EEGTools.Recorders.LiveAmpRecorder.Backends import Sawtooth as backend

import asyncio

# initialize the recorder and connect it
rec = Recorder(backend=backend.get_backend())
rec.connect()
print('-------Recorder Connected--------')


async def start_recording():

    rec.refresh()
    rec.start_recording()
    await asyncio.sleep(3)
    rec.refresh()


def set_event_with_offset(event_id, time):
    offset = 500 * time
    rec.refresh()
    rec.set_event(event_id, offset)
    print('-------Event set--------')


def get_latest_data_from_buffer():
    rec.refresh()
    data = rec.get_new_data()
    return data


def stop_recording(subject_id):
    rec.stop_recording()
    print('Recording has been stopped!')
    rec.disconnect()
    rec.save(file_prefix=f"subject_{subject_id}_online_raw", path=get_path('online_module_data'), description='Online Module Data Recording')
    rec.refresh()
    rec.clear()


def get_events() -> list:
    return rec.get_events()


def get_ch_names() -> list:
    return rec.get_names()
