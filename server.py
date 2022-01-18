import asyncio
import zmq.asyncio
from online_module import classify_label

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


async def action_based_on_request(repsocket, request_message, ctr):
    if request_message == 'RequestToStartRecording':
        await asyncio.sleep(2)
        await repsocket.send_string('enabled')
    elif request_message == 'RequestForClassifiedLabel':
        ctr += 1
        label = await classify_label(ctr)
        await repsocket.send_string(label)
    elif 'RequestToCorrectLabel' in request_message:
        label = request_message.split()[-1]
        await asyncio.sleep(2)
        await repsocket.send_string('Label corrected')
    elif request_message == 'RequestToRetrainModel':
        await asyncio.sleep(2)
        #TODO: Add the online training method file
        await repsocket.send_string('Retrained')
        print('response to retrain sent')
    elif request_message == 'Hello':
        await asyncio.sleep(2)
        await repsocket.send_string('World')
        print('response sent')
    elif request_message == 'Hello again':
        await asyncio.sleep(2)
        await repsocket.send_string('World Again')
        print('response sent')


def main():
    try:
        while True:
            ctr=0
            message = asyncio.run(get_request())
            asyncio.run(action_based_on_request(socket, message, ctr))
    except KeyboardInterrupt:
        print('User triggered exit')
        socket.close(linger=0)
        context.term()
        raise SystemExit


if __name__ == '__main__':
    main()






