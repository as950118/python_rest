import flask
from flask import request, jsonify
import sqlite3
import json

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    return '''<h1>Distant Reading Archive</h1> <p>A prototype API for distant reading of science fiction novels.</p>'''

#Users
#Getting all users
@app.route('/bookmarking/users', methods=['GET'])
def api_user_all():
    users_table = ['user_id', 'user_name']
    sql = 'SELECT * FROM Users ORDER BY user_id ASC;'
    conn = sqlite3.connect('books.db')
    cur = conn.cursor()
    res = cur.execute(sql).fetchall()
    users = [dict(zip(users_table, r)) for r in res]
    return jsonify({'count':len(res), 'users':users}), 200

#Adding one or more new user(s)
@app.route('/bookmarking', methods=['POST'])
def api_user_one():
    users_table = ['user_id', 'user_name']
    try:
        data = request.get_json(silent=True)
        user_ids = data['user_id']
        user_names = data['user_name']
    # 500 error
    except Exception as e:
        return jsonify({"reasons":[{"message":"Internal Server Error"}]}), 500
    sql = 'INSERT INTO Users(user_id, user_name) values(?, ?);'
    try:
        conn = sqlite3.connect('books.db')
        cur = conn.cursor()
        # if input data is more than one
        if str(type(user_ids)) == "<class 'list'>":
            res = []
            already_exists = []
            for idx in range(len(user_ids)):
                user_id = user_ids[idx]
                user_name = user_names[idx]
                try:
                    cur.execute(sql, (user_id, user_name,))
                    conn.commit()
                    res.append(dict(zip(users_table, [user_id, user_name])))
                except Exception as e:
                    print(e)
                    return jsonify({"reasons":[{"message":"User already exists"}]}), 400
        # else input data is one
        else:
            user_id = user_ids
            user_name = user_names
            cur.execute(sql, (user_id, user_name,))
            conn.commit()
            res = dict(zip(users_table, [user_id, user_name]))
        return jsonify(res), 201
    # 400 error
    except Exception as e:
        print(e)
        return jsonify({"reasons":[{"message":"User already exists"}]}), 400

#Deleting a user
@app.route('/bookmarking/<user_id>', methods=['DELETE'])
def api_user_del(user_id):
    sql = 'DELETE FROM Users WHERE user_id=?;'
    conn = sqlite3.connect('books.db')
    cur = conn.cursor()
    res = cur.execute('SELECT * FROM Users WHERE user_id=?;', (user_id,)).fetchall()
    if res:
        cur.execute(sql, (user_id,))
        conn.commit()
        return '', 204
    else:
        return jsonify({"reasons":[{"message":"User does not exists"}]}), 404

#Bookmarks
#Getting all bookmarks
@app.route('/bookmarking/bookmarks', methods=['GET'])
def api_bookmark_all():
    bookmarks_table = ['url', 'tags', 'text', 'user_id']
    sql = 'SELECT * FROM Bookmarks ORDER BY user_id ASC;'
    conn = sqlite3.connect('books.db')
    cur = conn.cursor()
    res = cur.execute(sql).fetchall()
    bookmarks = [dict(zip(bookmarks_table, r)) for r in res]
    return jsonify({'count':len(res), 'bookmarks':bookmarks}), 200

#Getting all bookmarks for a certain user
@app.route('/bookmarking/bookmarks/<user_id>', methods=['GET'])
def api_bookmark_certain(user_id):
    bookmarks_table = ['url', 'tags', 'text', 'user_id']
    sql = 'SELECT * FROM Bookmarks WHERE user_id=? ORDER BY url ASC;'
    try:
        conn = sqlite3.connect('books.db')
        cur = conn.cursor()
        res = cur.execute(sql, (user_id,)).fetchall()
        conn.commit()
        bookmarks = [dict(zip(bookmarks_table, r)) for r in res]
        return jsonify({'count':len(res), 'bookmarks':bookmarks}), 200
    # 404 error
    except Exception as e:
        print(e)
        return jsonify({"reasons":[{"message":"The user does not exist"}]}), 404

#Getting target bookmarks for a certain user
@app.route('/bookmarking/bookmarks/<user_id>/<path:url>', methods=['GET'])
def api_bookmark_target(user_id, url):
    bookmarks_table = ['url', 'tags', 'text', 'user_id']
    sql = 'SELECT * FROM Bookmarks WHERE user_id=? AND url=?;'
    conn = sqlite3.connect('books.db')
    cur = conn.cursor()
    res = cur.execute(sql, (user_id, url,)).fetchall()
    conn.commit()
    if not res:
        return jsonify({"reasons":[{"message":"The user does not exist"}]}), 404
    else:
        bookmarks = [dict(zip(bookmarks_table, r)) for r in res]
        return jsonify({'count':len(res), 'bookmarks':bookmarks}), 200

#Adding one or more bookmark(s) for a user
@app.route('/bookmarking/<user_id>/bookmarks', methods=['POST'])
def api_bookmark_add(user_id):
    bookmarks_table = ['url', 'tags', 'text', 'user_id']
    data = request.get_json(silent=True)
    # url & user_id is essential / tags & text is not essential
    if not user_id:
        return jsonify({"reasons": [{"message": "User does not exist"}]}), 404
    try:
        urls = data['url']
    except Exception as e:
        print(e)
        return jsonify({"reasons":[{"message":"Url does not exist"}]}), 500
    try:
        tagss = data['tags']
    except Exception as e:
        print(e)
        tagss = [[] for i in range(len(urls))]
    try:
        texts = data['text']
    except Exception as e:
        print(e)
        text = [[] for i in range(len(urls))]
    try:
        sql = 'INSERT INTO Bookmarks(url, tags, text, user_id) VALUES(?, ?, ?, ?);'
        conn = sqlite3.connect('books.db')
        cur = conn.cursor()
        # if input data is more than one
        if str(type(urls)) == "<class 'list'>":
            res = []
            already_exists = []
            for idx in range(len(urls)):
                url = urls[idx]
                tags = tagss[idx]
                text = texts[idx]
                try:
                    cur.execute(sql, (url, tags, text, user_id,))
                    conn.commit()
                    res.append([url, tags, text, user_id])
                except Exception as e:
                    print(e)
                    return jsonify({"reasons": [{"message": "User and url does already exist"}]}), 400
        # else input data is one
        else:
            url = urls
            tags = tagss
            text = texts
            try:
                cur.execute(sql, (url, tags, text, user_id,))
                conn.commit()
            except Exception as e:
                print(e)
                return jsonify({"reasons": [{"message": "User and url does already exist"}]}), 400
            # test
            sql = 'SELECT * FROM Bookmarks WHERE user_id=? AND url=?;'
            res = cur.execute(sql, (user_id, url,)).fetchall()
            conn.commit()
        bookmarks = [dict(zip(bookmarks_table, r)) for r in res]
        return jsonify({'count':len(res), 'bookmarks':bookmarks}), 201
    except Exception as e:
        print(e)
        return jsonify({"reasons":[{"message":"The user or url does not exist"}]}), 404

#Updating the title/tag(s) for a bookmarks for a target user
@app.route('/bookmarking/<user_id>/bookmarks/<path:url>', methods=['PUT'])
def api_bookmark_update_delete(user_id, url):
    bookmarks_table = ['url', 'tags', 'text', 'user_id']
    data = request.get_json(silent=True)
    # url & user_id is essential / tags & text is not essential
    if not user_id or not url:
        return jsonify({"reasons":[{"message":"Request is incorrect"}]}), 500
    try:
        tags = data['tags']
    except Exception as e:
        print(e)
        tags = ''
    try:
        text = data['text']
    except Exception as e:
        print(e)
        text = ''
    try:
        sql = 'UPDATE Bookmarks SET tags=?, text=? WHERE url=? AND user_id=?;'
        conn = sqlite3.connect('books.db')
        cur = conn.cursor()
        cur.execute(sql, (tags, text, url, user_id,))
        conn.commit()
        # test
        sql = 'SELECT * FROM Bookmarks WHERE user_id=? AND url=?;'
        res = cur.execute(sql, (user_id, url,)).fetchall()
        conn.commit()
        bookmarks = [dict(zip(bookmarks_table, r)) for r in res]
        return jsonify({'count':len(res), 'bookmarks':bookmarks}), 201
    except Exception as e:
        print(e)
        return jsonify({"reasons":[{"message":"The user or url does not exist"}]}), 404

#Deleting a bookmark for a target user
@app.route('/bookmarking/<user_id>/bookmarks/<path:url>', methods=['DELETE'])
def api_bookmark_del(user_id, url):
    try:
        conn = sqlite3.connect('books.db')
        cur = conn.cursor()
        sql = 'SELECT * FROM Bookmarks WHERE user_id=? AND url=?;'
        res = cur.execute(sql, (user_id, url,)).fetchall()
        conn.commit()
        if not res:
            return jsonify({"reasons": [{"message": "The user or bookmark does not exist"}]}), 404
        else:
            sql = 'DELETE FROM Bookmarks WHERE user_id=? AND url=?;'
            cur.execute(sql, (user_id, url,))
            conn.commit()
            return '', 204
    except Exception as e:
        print(e)
        return jsonify({"reasons":[{"message":"Request in incorrect"}]}), 500

@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404</h1><p>The resource could not be found.</p>", 404

@app.route('/bookmarking', methods=['GET'])
def api_filter():
    query_parameters = request.args
    url = query_parameters.get('url')
    tags = query_parameters.get('tags')
    text = query_parameters.get('text')
    user_id = query_parameters.get('user_id')

    query = "SELECT * FROM bookmarks WHERE"
    to_filter = []

    if url:
        query += ' url=? AND'
        to_filter.append(url)
    if tags:
        query += ' tags=? AND'
        to_filter.append(tags)
    if text:
        query += ' text=? AND'
        to_filter.append(text)
    if user_id:
        query += ' user_id=? AND'
        to_filter.append(user_id)
    if not (url or tags or text or user_id):
        return page_not_found(404)

    query = query[:-4] + ';'
 
    conn = sqlite3.connect('books.db')
    cur = conn.cursor()
    results = cur.execute(query, to_filter).fetchall()
    
    return jsonify(results)

app.run()