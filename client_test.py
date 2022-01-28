import zmq
import asyncio
import zmq.asyncio
import time

context = zmq.asyncio.Context()
socket = context.socket(zmq.REQ)
socket.connect('tcp://localhost:5555')


async def main():
        # await socket.send_multipart([b'Hello', b'again'])
        await socket.send(b'RequestToStartRecording')
        message = await socket.recv_string()
        print(f'Received reply : {message}')
        await asyncio.sleep(2)
        timer = time.time()
        await socket.send_multipart([timer, b'RequestToAddSpacekeyEvent'])
        message = await socket.recv_string()
        print(f'Received reply : {message}')
        await asyncio.sleep(6)
        await socket.send(b'RequestForClassifiedLabel')
        #  Get the reply.
        message = await socket.recv_string()
        print(f'Received reply : {message}')



if __name__ == '__main__':
    asyncio.run(main())
