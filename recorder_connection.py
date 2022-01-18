from EEGTools.Recorders.LiveAmpRecorder.liveamp_recorder import LiveAmpRecorder as Recorder

import asyncio

# initialize the recorder and connect it
rec = Recorder()
rec.connect()


async def start_recording():
    rec.start_recording()
    await asyncio.sleep(3)
    rec.refresh()


async def set_event_with_offset(event_id, offset):
    rec.set_event(event_id, offset)

def get_latest_data_from_buffer():
    rec.refresh()
    data = rec.get_new_data()
    return data