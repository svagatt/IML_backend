import zmq
from recorder_connection import start_recording
import asyncio

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind('tcp://*:5555')


def main():
    message = socket.recv_string()
    print(f'Request Received: {message}')
    asyncio.run(action_based_on_request(socket, message))


async def action_based_on_request(repsocket, message):
    if message == 'RequestToStartRecording':
        await start_recording()
        repsocket.send_string('enabled')

if __name__ == '__main__':
    main()
    input()
