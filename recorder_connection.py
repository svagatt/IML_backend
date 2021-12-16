from EEGTools.Recorders.LiveAmpRecorder.liveamp_recorder import LiveAmpRecorder as Recorder

import asyncio


async def start_recording():
    rec = Recorder()
    """ add subject id and other details on demographics"""
    subject_demographics = rec.subject_info_ui()
    rec.connect()
    rec.start_recording()
    await asyncio.sleep(5)
    rec.refresh()

