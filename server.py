import zmq
import asyncio

from recorder_connection import start_recording
from load_model_to_predict import get_label

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind('tcp://*:5555')


def main():
    message = socket.recv_string()
    print(f'Request Received: {message}')
    asyncio.run(action_based_on_request(socket, message))


async def action_based_on_request(repsocket, request_message):
    if request_message == 'RequestToStartRecording':
        await start_recording()
        repsocket.send_string('enabled')
    elif request_message == 'RequestForClassifiedLabel':
        # label = await get_label(sample)
        label = 'schraube'
        repsocket.send_string(label)


if __name__ == '__main__':
    main()
    input()
