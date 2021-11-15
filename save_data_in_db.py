from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import sys
from pprint import pprint
def main():
    # connect to mongodb
    try:
        print('--------Trying to connect to MongoDB--------')
        client = MongoClient('localhost', 8080)
        print('Connnected sucessfully')
        print(client)
    except ConnectionFailure, e:
        sys.strderr.write()
        db = client.admin
