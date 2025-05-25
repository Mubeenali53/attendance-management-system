import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://faceattendance-fe76b-default-rtdb.firebaseio.com/"
})

ref = db.reference('Students Data')

data = {
    "123":
        {
            "name": "xyz",
            "starting_year": 2017,
            "total_attendance": 7,
            "year": 3,
            "last_attendance_time": "2022-12-11 00:54:34"
        },
    "frame1":
        {
            "name": "abc",
            "starting_year": 2021,
            "total_attendance": 12,
            "year": 4,
            "last_attendance_time": "2022-12-11 00:54:34"
        },
    "frame2":
        {
            "name": "def",
            "starting_year": 2021,
            "total_attendance": 0,
            "year": 4,
            "last_attendance_time": "2022-12-11 00:54:34"
        },
}

for key, value in data.items():
    ref.child(key).set(value)
