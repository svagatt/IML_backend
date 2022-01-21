#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2021.2.3),
    on January 17, 2022, at 18:48
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

from __future__ import absolute_import, division

from psychopy import locale_setup
from psychopy import prefs
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
                   
from numpy.random import random, randint, normal, shuffle, choice as randchoice
from numpy.random import uniform
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard


# External imports

from EEGTools.Recorders.LiveAmpRecorder.liveamp_recorder import LiveAmpRecorder as Recorder
from EEGTools.Recorders.LiveAmpRecorder.Backends import Sawtooth as backend


folder_name = 'offline_module_data'
loop_part = 0
events_to_set = [] 


# a get_path def to create a dir or the get an existing dir
def get_path(directory_name):
    path = f'C:\\Users\\mash02-admin\\varsha_thesis\\ml_part\\machine_learning_part\\{directory_name}\\'
    return path


# stop recording and save the file
def stop_recording():
    rec.stop_recording()
    print('Recording has been stopped!')
    rec.disconnect()
    rec.set_event_dict(event_list)
    rec.save(file_prefix=f"subject_{expInfo['participant']}_raw", path=get_path(folder_name), description='Offline Module Data Recording', save_additional = True, subject_info = expInfo)
    rec.refresh()
    rec.clear()

def set_event(event_id):
    rec.refresh()
    rec.set_event(event_id)
    print(f'Event set: {event_id}')

    
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Store info about the experiment session
psychopyVersion = '2021.2.3'
expName = 'offline_module'  # from the Builder filename that created this script
expInfo = {'participant': ''}
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='C:\\Users\\mash02-admin\\varsha_thesis\\ml_part\\machine_learning_part\\psychopy\\offline_module.py',
    savePickle=True, saveWideText=False,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# events that occur during the recording
event_list = {'Schraube_start': 10, 'Platine_start': 20, 'Gehäuse_start': 30, 'Werkbank_start': 40, 'Fließband_start': 50, 'Boden_start': 60, 'Lege_start': 70, 'Halte_start': 80, 'Hebe_start': 90,
              'Schraube_end': 11, 'Platine_end': 21, 'Gehäuse_end': 31, 'Werkbank_end': 41, 'Fließband_end': 51, 'Boden_end': 61, 'Lege_end': 71, 'Halte_end': 81, 'Hebe_end': 91, 
              'Pause_start':19, 'Pause_end': 29, 'Space_key': 39, 'Mouse_click': 49, 'Dummy': 99}

# initialize recorder
rec = Recorder()
rec.connect()

# Start Code - component code to be run after the window creation

# Setup the Window
win = visual.Window(
    size=[1920, 1080], fullscr=False, screen=0, 
    winType='pyglet', allowGUI=False, allowStencil=False,
    monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='height')
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess
# Start Code - component code to be run after the window creation

# Setup eyetracking
ioDevice = ioConfig = ioSession = ioServer = eyetracker = None

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard()

# Initialize components for Routine "instr"
instrClock = core.Clock()
welcome_text = visual.TextStim(win=win, name='welcome_text',
    text="Willkommen beim Experiment. Dies ist das Offline-Modul, in dem Ihnen ein Wort angezeigt wird. Nachdem das Wort dargestellt wurde, erscheint ein Fixationskreuz. Sobald dieses erscheint müssen Sie sich so lange das Wort vorstellen, bis das Fixationskreuz wieder verschwindet. \n\n\nDieser Ablauf wird dreimal pro Wort, für mehrere Wörter, wiederholt. Wenn Sie diesen Text verstanden haben, drücken Sie die ’Leertaste' um fortzufahren",
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
space_key = keyboard.Keyboard()

# Initialize components for Routine "trial"
trialClock = core.Clock()
text = visual.TextStim(win=win, name='text',
    text='',
    font='Open Sans',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
before_cross_pause = visual.TextStim(win=win, name='before_cross_pause',
    text=None,
    font='Open Sans',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);
fix_cross = visual.ShapeStim(
    win=win, name='fix_cross', vertices='cross',
    size=(0.1, 0.1),
    ori=0.0, pos=(0, 0),
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-2.0, interpolate=True)
text_2 = visual.TextStim(win=win, name='text_2',
    text=None,
    font='Open Sans',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-3.0);
fix_cross2 = visual.ShapeStim(
    win=win, name='fix_cross2', vertices='cross',
    size=(0.1, 0.1),
    ori=0.0, pos=(0, 0),
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-4.0, interpolate=True)
text_3 = visual.TextStim(win=win, name='text_3',
    text=None,
    font='Open Sans',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-5.0);
fix_cross3 = visual.ShapeStim(
    win=win, name='fix_cross3', vertices='cross',
    size=(0.1, 0.1),
    ori=0.0, pos=(0, 0),
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-6.0, interpolate=True)
text_4 = visual.TextStim(win=win, name='text_4',
    text=None,
    font='Open Sans',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-7.0);
random_duration = 0

# Initialize components for Routine "pause"
pauseClock = core.Clock()
pause_text = visual.TextStim(win=win, name='pause_text',
    text="Jetzt gibt es 1 Minute Pause, wenn Sie keine Pause wollen, drücken Sie die ’Leertaste' um fortzufahren",
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
text_5 = visual.TextStim(win=win, name='text_5',
    text=str(round(pauseClock.getTime(),0)),
    font='Open Sans',
    pos=(0, -0.2), height=0.2, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);
space_key_1 = keyboard.Keyboard()

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# ------Prepare to start Routine "instr"-------
continueRoutine = True
# update component parameters for each repeat
space_key.keys = []
space_key.rt = []
_space_key_allKeys = []
# keep track of which components have finished
instrComponents = [welcome_text, space_key]
for thisComponent in instrComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
instrClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "instr"-------
while continueRoutine:
    # get current time
    t = instrClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=instrClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *welcome_text* updates
    if welcome_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        welcome_text.frameNStart = frameN  # exact frame index
        welcome_text.tStart = t  # local t and not account for scr refresh
        welcome_text.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(welcome_text, 'tStartRefresh')  # time at next scr refresh
        welcome_text.setAutoDraw(True)
    
    # *space_key* updates
    waitOnFlip = False
    if space_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        space_key.frameNStart = frameN  # exact frame index
        space_key.tStart = t  # local t and not account for scr refresh
        space_key.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(space_key, 'tStartRefresh')  # time at next scr refresh
        space_key.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(space_key.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(space_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if space_key.status == STARTED and not waitOnFlip:
        theseKeys = space_key.getKeys(keyList=['space'], waitRelease=False)
        _space_key_allKeys.extend(theseKeys)
        if len(_space_key_allKeys):
            space_key.keys = _space_key_allKeys[-1].name  # just the last key pressed
            space_key.rt = _space_key_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instrComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "instr"-------
for thisComponent in instrComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('welcome_text.started', welcome_text.tStartRefresh)
thisExp.addData('welcome_text.stopped', welcome_text.tStopRefresh)
# check responses
if space_key.keys in ['', [], None]:  # No response was made
    space_key.keys = None
thisExp.addData('space_key.keys',space_key.keys)
if space_key.keys != None:  # we had a response
    thisExp.addData('space_key.rt', space_key.rt)
thisExp.addData('space_key.started', space_key.tStartRefresh)
thisExp.addData('space_key.stopped', space_key.tStopRefresh)
thisExp.nextEntry()
# the Routine "instr" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
repeatx3 = data.TrialHandler(nReps=3.0, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='repeatx3')
thisExp.addLoop(repeatx3)  # add the loop to the experiment
thisRepeatx3 = repeatx3.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisRepeatx3.rgb)
if thisRepeatx3 != None:
    for paramName in thisRepeatx3:
        exec('{} = thisRepeatx3[paramName]'.format(paramName))

for thisRepeatx3 in repeatx3:
    # start recording after each break
    rec.start_recording()
    currentLoop = repeatx3
    loop_part += 1
    # abbreviate parameter names if possible (e.g. rgb = thisRepeatx3.rgb)
    if thisRepeatx3 != None:
        for paramName in thisRepeatx3:
            exec('{} = thisRepeatx3[paramName]'.format(paramName))
    
    # set up handler to look after randomisation of conditions etc
    condition_reps = data.TrialHandler(nReps=5.0, method='fullRandom', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('conditions.csv'),
        seed=None, name='condition_reps')
    thisExp.addLoop(condition_reps)  # add the loop to the experiment
    thisCondition_rep = condition_reps.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisCondition_rep.rgb)
    if thisCondition_rep != None:
        for paramName in thisCondition_rep:
            exec('{} = thisCondition_rep[paramName]'.format(paramName))
    
    for thisCondition_rep in condition_reps:
        currentLoop = condition_reps
        # abbreviate parameter names if possible (e.g. rgb = thisCondition_rep.rgb)
        if thisCondition_rep != None:
            for paramName in thisCondition_rep:
                exec('{} = thisCondition_rep[paramName]'.format(paramName))
        
        # ------Prepare to start Routine "trial"-------
        continueRoutine = True
        # update component parameters for each repeat
        text.setText(word)
        for key, value in event_list.items():
            if word in key:
                events_to_set.append(value)
                
        random_duration = round(uniform(1.1,2),1)
        
        # keep track of which components have finished
        trialComponents = [text, before_cross_pause, fix_cross, text_2, fix_cross2, text_3, fix_cross3, text_4]
        for thisComponent in trialComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        trialClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "trial"-------
        while continueRoutine:
            # get current time
            t = trialClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=trialClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text* updates
            if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text.frameNStart = frameN  # exact frame index
                text.tStart = t  # local t and not account for scr refresh
                text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                text.setAutoDraw(True)
            if text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    text.tStop = t  # not accounting for scr refresh
                    text.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(text, 'tStopRefresh')  # time at next scr refresh
                    text.setAutoDraw(False)
            
            # *before_cross_pause* updates
            if before_cross_pause.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                before_cross_pause.frameNStart = frameN  # exact frame index
                before_cross_pause.tStart = t  # local t and not account for scr refresh
                before_cross_pause.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(before_cross_pause, 'tStartRefresh')  # time at next scr refresh
                before_cross_pause.setAutoDraw(True)
            if before_cross_pause.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > before_cross_pause.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    before_cross_pause.tStop = t  # not accounting for scr refresh
                    before_cross_pause.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(before_cross_pause, 'tStopRefresh')  # time at next scr refresh
                    before_cross_pause.setAutoDraw(False)
            
            # *fix_cross* updates
            if fix_cross.status == NOT_STARTED and before_cross_pause.status == FINISHED:
                set_event(events_to_set[-2])
                # keep track of start time/frame for later
                fix_cross.frameNStart = frameN  # exact frame index
                fix_cross.tStart = t  # local t and not account for scr refresh
                fix_cross.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fix_cross, 'tStartRefresh')  # time at next scr refresh
                fix_cross.setAutoDraw(True)
            if fix_cross.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fix_cross.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    fix_cross.tStop = t  # not accounting for scr refresh
                    fix_cross.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(fix_cross, 'tStopRefresh')  # time at next scr refresh
                    fix_cross.setAutoDraw(False)
            
            # *text_2* updates
            if text_2.status == NOT_STARTED and fix_cross.status == FINISHED:
                set_event(events_to_set[-1])
                # keep track of start time/frame for later
                text_2.frameNStart = frameN  # exact frame index
                text_2.tStart = t  # local t and not account for scr refresh
                text_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
                text_2.setAutoDraw(True)
            if text_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_2.tStartRefresh + random_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    text_2.tStop = t  # not accounting for scr refresh
                    text_2.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(text_2, 'tStopRefresh')  # time at next scr refresh
                    text_2.setAutoDraw(False)
            
            # *fix_cross2* updates
            if fix_cross2.status == NOT_STARTED and text_2.status == FINISHED:
                set_event(events_to_set[-2])
                # keep track of start time/frame for later
                fix_cross2.frameNStart = frameN  # exact frame index
                fix_cross2.tStart = t  # local t and not account for scr refresh
                fix_cross2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fix_cross2, 'tStartRefresh')  # time at next scr refresh
                fix_cross2.setAutoDraw(True)
            if fix_cross2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fix_cross2.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    fix_cross2.tStop = t  # not accounting for scr refresh
                    fix_cross2.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(fix_cross2, 'tStopRefresh')  # time at next scr refresh
                    fix_cross2.setAutoDraw(False)
            
            # *text_3* updates
            if text_3.status == NOT_STARTED and fix_cross2.status == FINISHED:
                set_event(events_to_set[-1])                
                # keep track of start time/frame for later
                text_3.frameNStart = frameN  # exact frame index
                text_3.tStart = t  # local t and not account for scr refresh
                text_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
                text_3.setAutoDraw(True)
            if text_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_3.tStartRefresh + random_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    text_3.tStop = t  # not accounting for scr refresh
                    text_3.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(text_3, 'tStopRefresh')  # time at next scr refresh
                    text_3.setAutoDraw(False)
            
            # *fix_cross3* updates
            if fix_cross3.status == NOT_STARTED and text_3.status == FINISHED:
                set_event(events_to_set[-2])
                # keep track of start time/frame for later
                fix_cross3.frameNStart = frameN  # exact frame index
                fix_cross3.tStart = t  # local t and not account for scr refresh
                fix_cross3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fix_cross3, 'tStartRefresh')  # time at next scr refresh
                fix_cross3.setAutoDraw(True)
            if fix_cross3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fix_cross3.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    fix_cross3.tStop = t  # not accounting for scr refresh
                    fix_cross3.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(fix_cross3, 'tStopRefresh')  # time at next scr refresh
                    fix_cross3.setAutoDraw(False)
            
            # *text_4* updates
            if text_4.status == NOT_STARTED and fix_cross3.status == FINISHED:
                set_event(events_to_set[-1])
                # keep track of start time/frame for later
                text_4.frameNStart = frameN  # exact frame index
                text_4.tStart = t  # local t and not account for scr refresh
                text_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_4, 'tStartRefresh')  # time at next scr refresh
                text_4.setAutoDraw(True)
            if text_4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_4.tStartRefresh + random_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    text_4.tStop = t  # not accounting for scr refresh
                    text_4.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(text_4, 'tStopRefresh')  # time at next scr refresh
                    text_4.setAutoDraw(False)
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trialComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "trial"-------
        for thisComponent in trialComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        condition_reps.addData('text.started', text.tStartRefresh)
        condition_reps.addData('text.stopped', text.tStopRefresh)
        condition_reps.addData('before_cross_pause.started', before_cross_pause.tStartRefresh)
        condition_reps.addData('before_cross_pause.stopped', before_cross_pause.tStopRefresh)
        condition_reps.addData('fix_cross.started', fix_cross.tStartRefresh)
        condition_reps.addData('fix_cross.stopped', fix_cross.tStopRefresh)
        condition_reps.addData('text_2.started', text_2.tStartRefresh)
        condition_reps.addData('text_2.stopped', text_2.tStopRefresh)
        condition_reps.addData('fix_cross2.started', fix_cross2.tStartRefresh)
        condition_reps.addData('fix_cross2.stopped', fix_cross2.tStopRefresh)
        condition_reps.addData('text_3.started', text_3.tStartRefresh)
        condition_reps.addData('text_3.stopped', text_3.tStopRefresh)
        condition_reps.addData('fix_cross3.started', fix_cross3.tStartRefresh)
        condition_reps.addData('fix_cross3.stopped', fix_cross3.tStopRefresh)
        condition_reps.addData('text_4.started', text_4.tStartRefresh)
        condition_reps.addData('text_4.stopped', text_4.tStopRefresh)
        thisExp.addData('random_duration', random_duration)
        
        # the Routine "trial" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 5.0 repeats of 'condition_reps'
    
    # get names of stimulus parameters
    if condition_reps.trialList in ([], [None], None):
        params = []
    else:
        params = condition_reps.trialList[0].keys()
    # save data for this loop
    condition_reps.saveAsText(filename + 'condition_reps.csv', delim=',',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    # ------Prepare to start Routine "pause"-------
    continueRoutine = True
    routineTimer.add(60.000000)
    # update component parameters for each repeat
    space_key_1.keys = []
    space_key_1.rt = []
    _space_key_1_allKeys = []
    # keep track of which components have finished
    pauseComponents = [pause_text, text_5, space_key_1]
    for thisComponent in pauseComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    pauseClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "pause"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = pauseClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=pauseClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # *pause_text* updates
        if pause_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            pause_text.frameNStart = frameN  # exact frame index
            pause_text.tStart = t  # local t and not account for scr refresh
            pause_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(pause_text, 'tStartRefresh')  # time at next scr refresh
            pause_text.setAutoDraw(True)
            set_event(19)
        if pause_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > pause_text.tStartRefresh + 60-frameTolerance:
                # keep track of stop time/frame for later
                pause_text.tStop = t  # not accounting for scr refresh
                pause_text.frameNStop = frameN  # exact frame index
                win.timeOnFlip(pause_text, 'tStopRefresh')  # time at next scr refresh
                pause_text.setAutoDraw(False)
        
        # *text_5* updates
        if text_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_5.frameNStart = frameN  # exact frame index
            text_5.tStart = t  # local t and not account for scr refresh
            text_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_5, 'tStartRefresh')  # time at next scr refresh
            text_5.setAutoDraw(True)
        if text_5.status == STARTED:
            
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_5.tStartRefresh + 60-frameTolerance:
                # keep track of stop time/frame for later
                text_5.tStop = t  # not accounting for scr refresh
                text_5.frameNStop = frameN  # exact frame index
                win.timeOnFlip(text_5, 'tStopRefresh')  # time at next scr refresh
                text_5.setAutoDraw(False)
        if text_5.status == STARTED:  # only update if drawing
            text_5.setText(str(round(pauseClock.getTime(),0)), log=False)
        
        # *space_key_1* updates
        waitOnFlip = False
        if space_key_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            space_key_1.frameNStart = frameN  # exact frame index
            space_key_1.tStart = t  # local t and not account for scr refresh
            space_key_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(space_key_1, 'tStartRefresh')  # time at next scr refresh
            space_key_1.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(space_key_1.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(space_key_1.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if space_key_1.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > space_key_1.tStartRefresh + 60-frameTolerance:
                # keep track of stop time/frame for later
                space_key_1.tStop = t  # not accounting for scr refresh
                space_key_1.frameNStop = frameN  # exact frame index
                win.timeOnFlip(space_key_1, 'tStopRefresh')  # time at next scr refresh
                space_key_1.status = FINISHED
        if space_key_1.status == STARTED and not waitOnFlip:
            theseKeys = space_key_1.getKeys(keyList=['y', 'n', 'left', 'right', 'space'], waitRelease=False)
            _space_key_1_allKeys.extend(theseKeys)
            if len(_space_key_1_allKeys):
                space_key_1.keys = _space_key_1_allKeys[-1].name  # just the last key pressed
                space_key_1.rt = _space_key_1_allKeys[-1].rt
                set_event(39)
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in pauseComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "pause"-------
    for thisComponent in pauseComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    repeatx3.addData('pause_text.started', pause_text.tStartRefresh)
    repeatx3.addData('pause_text.stopped', pause_text.tStopRefresh)
    repeatx3.addData('text_5.started', text_5.tStartRefresh)
    repeatx3.addData('text_5.stopped', text_5.tStopRefresh)
    # check responses
    if space_key_1.keys in ['', [], None]:  # No response was made
        space_key_1.keys = None
    repeatx3.addData('space_key_1.keys',space_key_1.keys)
    if space_key_1.keys != None:  # we had a response
        repeatx3.addData('space_key_1.rt', space_key_1.rt)
    repeatx3.addData('space_key_1.started', space_key_1.tStartRefresh)
    repeatx3.addData('space_key_1.stopped', space_key_1.tStopRefresh)
    set_event(29)

# completed 3.0 repeats of 'repeatx3'

# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()
stop_recording()
# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
