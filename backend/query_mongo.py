from pymongo import MongoClient
client = MongoClient("mongodb://127.0.0.1:27020/")
db = client["vinh"]
print("Collections:", db.list_collection_names())
for doc in db["global_settings"].find():
    print(doc)
for doc in db["settings"].find():
    print(doc)
for doc in db["users"].find():
    print(doc)
