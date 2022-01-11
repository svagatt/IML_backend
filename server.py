import asyncio
import zmq.asyncio

context = zmq.asyncio.Context()
socket = context.socket(zmq.REP)
socket.bind('tcp://*:5555')


async def get_request():
    message = await socket.recv_string()
    print(f'Request Received: {message}')
    return message


async def action_based_on_request(repsocket, request_message):
    if request_message == 'RequestToStartRecording':
        await asyncio.sleep(2)
        await repsocket.send_string('enabled')
    elif request_message == 'RequestForClassifiedLabel':
        # label = await get_label(sample)
        label = 'schraube'
        await repsocket.send_string(label)
    elif request_message == 'Hello':
        await asyncio.sleep(2)
        await repsocket.send_string('World')
        print('response sent')
    elif request_message == 'RequestToRetrainModel':
        await asyncio.sleep(5)
        #TODO: Add the online training method file
        await repsocket.send_string('Retrained')
        print('response to retrain sent')


async def main():
    try:
        while True:
            message = await get_request()
            await action_based_on_request(socket, message)
    except KeyboardInterrupt:
        print('User triggered exit')
        socket.close(linger=0)
        context.term()
        raise SystemExit

if __name__ == '__main__':
    asyncio.run(main())






