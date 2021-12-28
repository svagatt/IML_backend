#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2021.2.3),
    on Thu Dec 23 16:33:06 2021
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
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# imports related to recorder
#from recorder_connection import initialize_recorder

#from read_save_data_files import get_path


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
                                 originPath='/Users/MindPalace/thesis_implementation/ml_classification/offline_module.py',
                                 savePickle=True, saveWideText=False,
                                 dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename + '.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# events that occur during the recording
event_list = {
                 'Schraube': 1,
                 'Platine': 2,
                 'Gehaüse': 3,
                 'Werkbank': 4,
                 'Fließband': 5,
                 'Boden': 6,
                 'Lege': 7,
                 'Halte': 8,
                 'Hebe': 9,
             },

# initialize recorder for the offline module
# rec = initialize_recorder(True)

# set event dict
# rec.set_event_dict(event_list)

# Start Code - component code to be run after the window creation

# Setup the Window
win = visual.Window(
    size=[1440, 900], fullscr=True, screen=0,
    winType='pyglet', allowGUI=False, allowStencil=False,
    monitor='testMonitor', color=[0, 0, 0], colorSpace='rgb',
    blendMode='avg', useFBO=True,
    units='height')
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess

# Setup eyetracking
ioDevice = ioConfig = ioSession = ioServer = eyetracker = None

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard()

# Initialize components for Routine "instr"
instrClock = core.Clock()
welcome_text = visual.TextStim(win=win, name='welcome_text',
                               text="Willkommen beim Experiment. Dies ist das Offline-Modul, in dem Ihnen ein Wort angezeigt wird und jedes Mal, wenn ein Fixationskreuz erscheint und sobald es verschwindet, müssen Sie sich das Wort vorstellen, dies würde dreimal pro Wort passieren, bevor das neue Wort angezeigt wird, wenn Sie Habe diesen Text verstanden, drücke 'Leertaste' um fortzufahren",
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
fix_cross = visual.ShapeStim(
    win=win, name='fix_cross', vertices='cross',
    size=(0.5, 0.5),
    ori=0.0, pos=(0, 0),
    lineWidth=1.0, colorSpace='rgb', lineColor='white', fillColor='white',
    opacity=None, depth=-1.0, interpolate=True)
text_2 = visual.TextStim(win=win, name='text_2',
                         text=None,
                         font='Open Sans',
                         pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0,
                         color='white', colorSpace='rgb', opacity=None,
                         languageStyle='LTR',
                         depth=-2.0);
fix_cross2 = visual.ShapeStim(
    win=win, name='fix_cross2', vertices='cross',
    size=(0.5, 0.5),
    ori=0.0, pos=(0, 0),
    lineWidth=1.0, colorSpace='rgb', lineColor='white', fillColor='white',
    opacity=None, depth=-3.0, interpolate=True)
text_3 = visual.TextStim(win=win, name='text_3',
                         text=None,
                         font='Open Sans',
                         pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0,
                         color='white', colorSpace='rgb', opacity=None,
                         languageStyle='LTR',
                         depth=-4.0);
fix_cross3 = visual.ShapeStim(
    win=win, name='fix_cross3', vertices='cross',
    size=(0.5, 0.5),
    ori=0.0, pos=(0, 0),
    lineWidth=1.0, colorSpace='rgb', lineColor='white', fillColor='white',
    opacity=None, depth=-5.0, interpolate=True)
text_4 = visual.TextStim(win=win, name='text_4',
                         text=None,
                         font='Open Sans',
                         pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0,
                         color='white', colorSpace='rgb', opacity=None,
                         languageStyle='LTR',
                         depth=-6.0);

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
    if welcome_text.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
        # keep track of start time/frame for later
        welcome_text.frameNStart = frameN  # exact frame index
        welcome_text.tStart = t  # local t and not account for scr refresh
        welcome_text.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(welcome_text, 'tStartRefresh')  # time at next scr refresh
        welcome_text.setAutoDraw(True)

    # *space_key* updates
    waitOnFlip = False
    if space_key.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
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
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape", "q"]):
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
thisExp.addData('space_key.keys', space_key.keys)
if space_key.keys != None:  # we had a response
    thisExp.addData('space_key.rt', space_key.rt)
    # rec.start_recording()
thisExp.addData('space_key.started', space_key.tStartRefresh)
thisExp.addData('space_key.stopped', space_key.tStopRefresh)
thisExp.nextEntry()
# the Routine "instr" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
trials = data.TrialHandler(nReps=15.0, method='fullRandom',
                           extraInfo=expInfo, originPath=-1,
                           trialList=data.importConditions('conditions.csv'),
                           seed=None, name='trials')
thisExp.addLoop(trials)  # add the loop to the experiment
thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
if thisTrial != None:
    for paramName in thisTrial:
        exec('{} = thisTrial[paramName]'.format(paramName))

for thisTrial in trials:
    currentLoop = trials
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            exec('{} = thisTrial[paramName]'.format(paramName))

    # ------Prepare to start Routine "trial"-------
    continueRoutine = True
    routineTimer.add(10.250000)
    # update component parameters for each repeat
    text.setText(word)
    # for key, value in event_list:
    #     if word == key:
    #         rec.set_event(value)


    # keep track of which components have finished
    trialComponents = [text, fix_cross, text_2, fix_cross2, text_3, fix_cross3, text_4]
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
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = trialClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=trialClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame

        # *text* updates
        if text.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            text.setAutoDraw(True)
        if text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text.tStartRefresh + 2.0 - frameTolerance:
                # keep track of stop time/frame for later
                text.tStop = t  # not accounting for scr refresh
                text.frameNStop = frameN  # exact frame index
                win.timeOnFlip(text, 'tStopRefresh')  # time at next scr refresh
                text.setAutoDraw(False)

        # *fix_cross* updates
        if fix_cross.status == NOT_STARTED and tThisFlip >= 2.0 - frameTolerance:
            # keep track of start time/frame for later
            fix_cross.frameNStart = frameN  # exact frame index
            fix_cross.tStart = t  # local t and not account for scr refresh
            fix_cross.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fix_cross, 'tStartRefresh')  # time at next scr refresh
            fix_cross.setAutoDraw(True)
        if fix_cross.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > fix_cross.tStartRefresh + 1 - frameTolerance:
                # keep track of stop time/frame for later
                fix_cross.tStop = t  # not accounting for scr refresh
                fix_cross.frameNStop = frameN  # exact frame index
                win.timeOnFlip(fix_cross, 'tStopRefresh')  # time at next scr refresh
                fix_cross.setAutoDraw(False)

        # *text_2* updates
        if text_2.status == NOT_STARTED and tThisFlip >= 3 - frameTolerance:
            # keep track of start time/frame for later
            text_2.frameNStart = frameN  # exact frame index
            text_2.tStart = t  # local t and not account for scr refresh
            text_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
            text_2.setAutoDraw(True)
        if text_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_2.tStartRefresh + 1.75 - frameTolerance:
                # keep track of stop time/frame for later
                text_2.tStop = t  # not accounting for scr refresh
                text_2.frameNStop = frameN  # exact frame index
                win.timeOnFlip(text_2, 'tStopRefresh')  # time at next scr refresh
                text_2.setAutoDraw(False)

        # *fix_cross2* updates
        if fix_cross2.status == NOT_STARTED and tThisFlip >= 4.75 - frameTolerance:
            # keep track of start time/frame for later
            fix_cross2.frameNStart = frameN  # exact frame index
            fix_cross2.tStart = t  # local t and not account for scr refresh
            fix_cross2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fix_cross2, 'tStartRefresh')  # time at next scr refresh
            fix_cross2.setAutoDraw(True)
        if fix_cross2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > fix_cross2.tStartRefresh + 1 - frameTolerance:
                # keep track of stop time/frame for later
                fix_cross2.tStop = t  # not accounting for scr refresh
                fix_cross2.frameNStop = frameN  # exact frame index
                win.timeOnFlip(fix_cross2, 'tStopRefresh')  # time at next scr refresh
                fix_cross2.setAutoDraw(False)

        # *text_3* updates
        if text_3.status == NOT_STARTED and tThisFlip >= 5.75 - frameTolerance:
            # keep track of start time/frame for later
            text_3.frameNStart = frameN  # exact frame index
            text_3.tStart = t  # local t and not account for scr refresh
            text_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
            text_3.setAutoDraw(True)
        if text_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_3.tStartRefresh + 1.75 - frameTolerance:
                # keep track of stop time/frame for later
                text_3.tStop = t  # not accounting for scr refresh
                text_3.frameNStop = frameN  # exact frame index
                win.timeOnFlip(text_3, 'tStopRefresh')  # time at next scr refresh
                text_3.setAutoDraw(False)

        # *fix_cross3* updates
        if fix_cross3.status == NOT_STARTED and tThisFlip >= 7.5 - frameTolerance:
            # keep track of start time/frame for later
            fix_cross3.frameNStart = frameN  # exact frame index
            fix_cross3.tStart = t  # local t and not account for scr refresh
            fix_cross3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fix_cross3, 'tStartRefresh')  # time at next scr refresh
            fix_cross3.setAutoDraw(True)
        if fix_cross3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > fix_cross3.tStartRefresh + 1.0 - frameTolerance:
                # keep track of stop time/frame for later
                fix_cross3.tStop = t  # not accounting for scr refresh
                fix_cross3.frameNStop = frameN  # exact frame index
                win.timeOnFlip(fix_cross3, 'tStopRefresh')  # time at next scr refresh
                fix_cross3.setAutoDraw(False)

        # *text_4* updates
        if text_4.status == NOT_STARTED and tThisFlip >= 8.5 - frameTolerance:
            # keep track of start time/frame for later
            text_4.frameNStart = frameN  # exact frame index
            text_4.tStart = t  # local t and not account for scr refresh
            text_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_4, 'tStartRefresh')  # time at next scr refresh
            text_4.setAutoDraw(True)
        if text_4.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_4.tStartRefresh + 1.75 - frameTolerance:
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
    trials.addData('text.started', text.tStartRefresh)
    trials.addData('text.stopped', text.tStopRefresh)
    trials.addData('fix_cross.started', fix_cross.tStartRefresh)
    trials.addData('fix_cross.stopped', fix_cross.tStopRefresh)
    trials.addData('text_2.started', text_2.tStartRefresh)
    trials.addData('text_2.stopped', text_2.tStopRefresh)
    trials.addData('fix_cross2.started', fix_cross2.tStartRefresh)
    trials.addData('fix_cross2.stopped', fix_cross2.tStopRefresh)
    trials.addData('text_3.started', text_3.tStartRefresh)
    trials.addData('text_3.stopped', text_3.tStopRefresh)
    trials.addData('fix_cross3.started', fix_cross3.tStartRefresh)
    trials.addData('fix_cross3.stopped', fix_cross3.tStopRefresh)
    trials.addData('text_4.started', text_4.tStartRefresh)
    trials.addData('text_4.stopped', text_4.tStopRefresh)
    thisExp.nextEntry()

# completed 15.0 repeats of 'trials'

# get names of stimulus parameters
if trials.trialList in ([], [None], None):
    params = []
else:
    params = trials.trialList[0].keys()
# save data for this loop
trials.saveAsText(filename + 'trials.csv', delim=',',
                  stimOut=params,
                  dataOut=['n', 'all_mean', 'all_std', 'all_raw'])
# rec.disconnect()
# rec.save(path=get_path('offline_module_data'), description='Offline Module Data Recording', subject_info=expInfo['participant'])


# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
