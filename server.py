import zmq
import asyncio
import time


context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind('tcp://*:5555')


async def main():
    while True:
        message = socket.recv()
        print(f'Request Received: {message}')
        await asyncio.sleep(2)
        action_based_on_request(socket, message)


def action_based_on_request(socket, message):
    if message is b'RequestToStartRecording':
        socket.send(b'enabled')

if __name__ == '__main__':
    asyncio.run(main())
    input()
