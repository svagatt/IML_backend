import zmq
import asyncio
import zmq.asyncio

context = zmq.asyncio.Context()
socket = context.socket(zmq.REQ)
socket.connect('tcp://localhost:5555')


async def main():
    for request in range(10):
        print(f'Sending request{request}')
        # await socket.send_multipart([b'Hello', b'again'])
        await socket.send(b'Hello')
        await asyncio.sleep(2)

        #  Get the reply.
        message = await socket.recv_string()
        print(f'Received reply {request}: {message}')

if __name__ == '__main__':
    asyncio.run(main())
