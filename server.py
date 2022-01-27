import asyncio
import zmq.asyncio
import time

from online_module import classify_label, set_right_label, retrain_model
from db_connections import close_database
from recorder_connection import start_recording, set_event_with_offset

context = zmq.asyncio.Context()
socket = context.socket(zmq.REP)
socket.bind('tcp://*:5555')
# windows asyncio warning trigger
# asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


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
        await start_recording()
        await repsocket.send_string('enabled')
    elif request_message == 'RequestForClassifiedLabel':
        label = await classify_label()
        await repsocket.send_string(label)
    elif 'RequestToCorrectLabel' in request_message:
        label = request_message.split()[-1]
        await set_right_label(label)
        await repsocket.send_string('LabelCorrected')
    elif request_message == 'RequestToRetrainModel':
        await retrain_model()
        await repsocket.send_string('Retrained')
        print('response to retrain sent')
    elif 'RequestToAddSpacekeyEvent' in request_message:
        start_time = float(request_message.split()[-2])
        end_time = time.time()
        duration = end_time - start_time
        # duration = 0
        print(f'Elapsed Duration: {duration}')
        await set_event_with_offset(39, -duration)
        await set_event_with_offset(99, -duration+2)
        await repsocket.send_string('EventAdded')


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





