import asyncio
import zmq.asyncio

context = zmq.asyncio.Context()
socket = context.socket(zmq.REP)
socket.bind('tcp://*:5555')


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
        # label = await get_label(sample)
        label = 'schraube'
        await repsocket.send_string(label)
    elif request_message == 'Hello':
        await asyncio.sleep(2)
        await repsocket.send_string('World')
        print('response sent')
    elif request_message == 'Hello again':
        await asyncio.sleep(2)
        await repsocket.send_string('World Again')
        print('response sent')
    elif 'RequestToCorrectLabel' in request_message:
        label = request_message.split()[-1]
        await asyncio.sleep(2)
        await repsocket.send_string('Label corrected')
    elif request_message == 'RequestToRetrainModel':
        await asyncio.sleep(2)
        #TODO: Add the online training method file
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






