from EEGTools.Recorders.LiveAmpRecorder.liveamp_recorder import LiveAmpRecorder as Recorder

import asyncio


""" enable this only during offline module so the demographics are stored once when the offline mode is running"""


def initialize_recorder(is_offline: bool):
    rec = Recorder()
    if is_offline:
        rec.subject_info_ui()
    rec.connect()
    return rec


async def start_recording(is_offline: bool):
    rec = initialize_recorder(is_offline)
    rec.start_recording()
    await asyncio.sleep(5)
    rec.refresh()

