from EEGTools.Recorders.LiveAmpRecorder.liveamp_recorder import LiveAmpRecorder as Recorder
from read_save_data_files import get_path
from EEGTools.Recorders.LiveAmpRecorder.Backends import Sawtooth as backend
from db_connections import save_time_when_refreshed, get_recording_start_time, get_latest_refresh_times

import asyncio
import numpy as np

# initialize the recorder and connect it
rec = Recorder(backend=backend.get_backend())
rec.connect()
print('-------Recorder Connected--------')


async def start_recording():
    rec.clear()
    rec.start_recording()
    await save_time_when_refreshed('start_time')
    await asyncio.sleep(3)
    rec.refresh()


async def set_event_with_offset(event_id, time):
    offset = 500 * time
    rec.refresh()
    await save_time_when_refreshed('refresh_buffer')
    rec.set_event(event_id, offset)
    print('-------Event set--------')


async def get_latest_data_from_buffer():
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


def get_events() -> np.ndarray:
    return rec.get_events()


def get_ch_names() -> list:
    return rec.get_names()


async def get_only_related_events()-> list:
    all_events = [{'list': event} for event in get_events()]
    events = [event for event in all_events[-2:]]
    related_events = [[event[0] - get_elapsed_time_sample_points(), event[1], event[2]] for event in events]
    return related_events


async def get_elapsed_time_sample_points() -> int:
    st = await get_recording_start_time()
    lbt = await get_latest_refresh_times()
    # lbt1 = time_points[0]
    # lbt2 = time_points[1]
    st_to_end_time = lbt - st
    # buf_to_buf_time = lbt1 - lbt2
    # print(round(st_to_end_time, 2)*500, round(buf_to_buf_time, 2)*500)
    return round(st_to_end_time, 2)*500
