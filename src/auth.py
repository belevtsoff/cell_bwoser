from hashlib import md5
import csv

def get_users():
    with file('users.db') as fid:
        users = dict(csv.reader(fid))
    return users

def encrypt_pw(pw):
    return md5(pw).hexdigest()