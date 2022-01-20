import asyncio
import zmq.asyncio
from online_module import classify_label, find_event_id_of_label, set_right_label, retrain_model
from recorder_connection import start_recording, set_event_with_offset
from codetiming import Timer

context = zmq.asyncio.Context()
socket = context.socket(zmq.REP)
socket.bind('tcp://*:5555')
# windows asyncio warning trigger
# asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
timer = Timer()
is_timer_running = False


def start_timer():
    timer.start()
    is_timer_running = True


def stop_timer():
    time = timer.stop()
    is_timer_running = False
    return time


async def get_request():
    messages = await socket.recv_multipart()
    messages_str = []
    for message in messages:
        messages_str.append(message.decode('utf-8'))
        print(f'Request Received: {message}')
    complete_message =' '.join(messages_str)
    print(complete_message)
    return complete_message


async def action_based_on_request(repsocket, request_message):
    if request_message == 'RequestToStartRecording':
        await asyncio.sleep(2)
        await repsocket.send_string('enabled')
    elif request_message == 'RequestForClassifiedLabel':
        set_event_with_offset(99, 3)
        if not is_timer_running:
            timer.start()
        label = await classify_label()
        await repsocket.send_string(label)
    elif 'RequestToCorrectLabel' in request_message:
        label = request_message.split()[-1]
        elapsed_time = timer.stop()
        set_event_with_offset(find_event_id_of_label(label), int(round(elapsed_time, 0))+3)
        await set_right_label(label)
        await repsocket.send_string('LabelCorrected')
    elif 'RequestToSetLabel' in request_message:
        label = request_message.split()[-1]
        elapsed_time = timer.stop()
        set_event_with_offset(find_event_id_of_label(label), int(round(elapsed_time, 0))+3)
        await set_right_label(label)
        await repsocket.send_string('LabelSet')
    elif request_message == 'RequestToRetrainModel':
        await retrain_model()
        await repsocket.send_string('Retrained')
        print('response to retrain sent')


def main():
    try:
        while True:
            message = asyncio.run(get_request())
            asyncio.run(action_based_on_request(socket, message))
    except KeyboardInterrupt:
        print('User triggered exit')
        socket.close(linger=0)
        context.term()
        raise SystemExit


if __name__ == '__main__':
    main()





