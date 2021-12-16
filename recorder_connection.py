from EEGTools.Recorders.LiveAmpRecorder.liveamp_recorder import LiveAmpRecorder as Recorder

import asyncio

rec = Recorder()
""" add subject id and other details on demographics"""
subject_demographics = rec.subject_info_ui()
rec.connect()

async def start_recording():
    rec.start_recording()
    await asyncio.sleep(3)
    await rec.refresh()

