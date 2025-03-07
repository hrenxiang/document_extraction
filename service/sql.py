import os
import sqlite3

# 连接到 Chroma 的 SQLite 数据库
current_path = os.path.abspath(__file__)
parent_path1 = os.path.dirname(current_path)
parent_path = os.path.dirname(parent_path1)


def init_conversation_messages():
    conn = sqlite3.connect(f'{parent_path}/chroma_db/chroma.sqlite3')
    cursor = conn.cursor()

    # 检查表是否存在
    cursor.execute('''
    SELECT name FROM sqlite_master WHERE type='table' AND name='conversation_messages';
    ''')
    table_exists = cursor.fetchone()

    # 如果表不存在，则创建新表
    if not table_exists:
        cursor.execute('''
        CREATE TABLE conversation_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            parent_id TEXT,
            message_id TEXT,
            qa_id TEXT,
            qa_type TEXT,
            message_content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        ''')
        conn.commit()
        print("消息表已创建！")
    else:
        print("消息表已存在，无需创建。")

    # 关闭数据库连接
    cursor.close()
    conn.close()


def insert_into_conversation_messages(user_id: str, session_id: str, parent_id: int, message_id: str, qa_id: int,
                                      qa_type: str,
                                      message_content: str):
    # 连接到 Chroma 的 SQLite 数据库
    print(f'{parent_path}/chroma_db/chroma.sqlite3')
    conn = sqlite3.connect(f'{parent_path}/chroma_db/chroma.sqlite3')
    cursor = conn.cursor()

    # 插入一条新消息到 conversation_messages 表
    try:
        cursor.execute('''
        INSERT INTO conversation_messages 
        (user_id, session_id, parent_id, message_id, qa_id, qa_type, message_content) 
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, session_id, parent_id, message_id, qa_id, qa_type, message_content))

        conn.commit()
        print("数据插入成功！")

    except sqlite3.IntegrityError as e:
        print("插入数据失败：", e)

    # 关闭数据库连接
    cursor.close()
    conn.close()


def query_session_history(user_id: str, session_id: str):
    # 连接到 Chroma 的 SQLite 数据库
    conn = sqlite3.connect(f'{parent_path}/chroma_db/chroma.sqlite3')
    cursor = conn.cursor()

    # 执行查询
    cursor.execute('''
    SELECT
    q.qa_id AS qa_id,
    q.qa_type,
    q.user_id,
    q.session_id,
    q.message_content AS question,
    (
        SELECT a.message_content
        FROM conversation_messages a
        WHERE a.qa_id = q.qa_id
          AND a.qa_type = 'answer'
        ORDER BY a.timestamp DESC
        LIMIT 1
    ) AS answer
    FROM
        conversation_messages q
    WHERE
        q.qa_type = 'question'
        AND q.user_id = '123'
        AND q.session_id = 'huangrx'
    ORDER BY
        q.timestamp ASC;
    ''')

    fetchall = cursor.fetchall()

    # 格式化结果为对话树
    conversation_tree = [
        {'question': row[1], 'answer': row[2]}
        for row in fetchall if row[1] and row[2]
    ]

    # 关闭数据库连接
    cursor.close()
    conn.close()

    return conversation_tree


def query_next_qa_id(user_id: str, session_id: str):
    # 连接到 Chroma 的 SQLite 数据库
    print(f'{parent_path}/chroma_db/chroma.sqlite3')
    conn = sqlite3.connect(f'{parent_path}/chroma_db/chroma.sqlite3')
    cursor = conn.cursor()

    # 执行查询
    cursor.execute('''
    SELECT MAX(CAST(SUBSTR(qa_id, 1) AS INTEGER)) FROM conversation_messages
    WHERE 
    user_id = ?
        AND session_id = ?
    ORDER BY 
        timestamp ASC;
    ''', (user_id, session_id,))

    max_qa_id = cursor.fetchone()[0]

    # 关闭数据库连接
    cursor.close()
    conn.close()
    return f'qa{(max_qa_id or 0) + 1}'


def query_top_message_id(user_id: str, session_id: str):
    # 连接到 Chroma 的 SQLite 数据库
    conn = sqlite3.connect(f'{parent_path}/chroma_db/chroma.sqlite3')
    cursor = conn.cursor()

    # 执行查询
    cursor.execute('''
    SELECT MAX(CAST(SUBSTR(qa_id, 1) AS INTEGER)) FROM conversation_messages
    WHERE 
        q.user_id = ?
        AND q.session_id = ?
    ORDER BY 
        q.timestamp ASC;
    ''', (user_id, session_id,))

    max_qa_id = cursor.fetchone()[0]

    # 关闭数据库连接
    cursor.close()
    conn.close()
    return f'qa{(max_qa_id or 0) + 1}'
