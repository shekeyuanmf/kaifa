# coding=utf-8
from flask import Flask, jsonify, request, make_response
import pandas as pd
import numpy as np
import json
# from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import pymysql
from sqlalchemy import create_engine

file = []
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import sys
import json

app = Flask(__name__)


# @app.route('/todo/api/v1.0/tasks', methods=['POST'])
# @app.route('/',methods=['GET','POST'])
# def knn():
#     #!/usr/bin/env python
#     # coding: utf-8
#     #默认数据库表为n列，前n-1列为特征列，最后一列为标签列。
#     #默认数据表是iris表，有5列
#     #已设置默认的远程数据库
#     #输出一个返回值，为模型的准确率
#     import numpy as np
#     import pandas as pd
#     #from sklearn.datasets import load_iris
#     from sklearn.neighbors import KNeighborsClassifier
#     import pymysql
#     from sqlalchemy import create_engine
#     from sklearn.model_selection import train_test_split
#     # from sklearn.model_selection import
#     user='root'
#     psw='bigdata@analysis'
#     host='10.110.10.15'
#     port=3306
#     db_name='test'
#     table='iris'
#     #连接数据库
#     #sql=json.loads(db)
#     # sql=request.json
#     # print(sql)
#     # user=sql['user']
#     # psw=sql['psw']
#     # host=sql['host']
#     # port=int(sql['port'])
#     # db_name=sql['database']
#     # table=sql['table']
#     neighbors=3
#     engine = create_engine('mysql+pymysql://{0}:{1}@{2}:{3}/{4}?charset=utf8'.format(user,psw,host,port,db_name))
#     df=pd.read_sql('select * from {} '.format(table),engine)
#     #取特征列和标签列
#     x=df.values[:,:-1]
#     y=df.values[:,-1]
#     #建模
#     knn=KNeighborsClassifier(n_neighbors=neighbors)
#
#     x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
#
#     model=knn.fit(x_train,y_train)
#     #输出
#     score=model.score(x_test,y_test)
#     pre=model.predict(x_test)
#     # 打印输出测试
#     #print(neighbors)
#     # print(score)
#     #print(score)
#     return jsonify({'score':score})

@app.route('/resource', methods=['POST', 'GET'])
# 作用:获取原始数据集
# 传参：无参数
# 返回值：json ，DataFrame形式
def resorce():
    user = 'root'
    psw = 'bigdata@analysis'
    host = '10.110.10.15'
    port = 3306
    db_name = 'test'
    table = 'iris'
    import numpy as np
    import pandas as pd
    # from sklearn.datasets import load_iris
    from sklearn.neighbors import KNeighborsClassifier
    import pymysql
    from sqlalchemy import create_engine
    from sklearn.model_selection import train_test_split
    engine = create_engine('mysql+pymysql://{0}:{1}@{2}:{3}/{4}?charset=utf8'.format(user, psw, host, port, db_name))
    df = pd.read_sql('select * from {} '.format(table), engine)
    df_json = df.to_json()
    rsp = make_response(df_json)
    rsp.mimetype = 'application/json'
    return rsp


@app.route('/split', methods=['POST'])
# 作用:划分数据集为训练集和测试集
# 传参：df的json形式
# 返回值：json ，键'train'对应训练集，键'test'对应测试集
def split():
    data = json.loads(request.get_data())
    df = pd.DataFrame(data)
    x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], shuffle=1)
    df_train = pd.concat([x_train, y_train], axis=1)
    df_test = pd.concat([x_test, y_test], axis=1)
    dict_df = {'train': df_train.to_json(), 'test': df_test.to_json()}
    return jsonify(dict_df)


@app.route('/model', methods=['POST'])
# 作用:利用训练集建模
# 传参：训练集，模型参数
# 返回值：json ，模型的参数
def model():
    data = json.loads(request.get_data())
    df = pd.DataFrame(data)
    lr = LogisticRegression()
    lr.fit(df.values[:, :-1], df.values[:, -1])
    model_dec = lr.get_params()
    global file
    file = joblib.dump(lr, 'lr.pkl')
    dict_df = {'model': model_dec, 'file': file}
    return jsonify(model_dec)


@app.route('/pre', methods=['POST'])
# 作用:利用模型验证
# 传参：测试集，模型
# 返回值：json ，模型的
def pre():
    data = json.loads(request.get_data())
    df = pd.DataFrame(data)
    global file
    print(file)
    model = joblib.load(file[0])
    print(model)
    # lr=LogisticRegression()
    # lr.fit(df.values[:,:-1],df.values[:,-1])
    # pre=lr.predict(df.values[:,:-1])
    # pre=np.array([2, 2, 2, 1, 1, 2, 2, 0, 0, 2, 2, 2, 1, 1, 1, 2, 2, 0, 1, 1, 0, 0,1, 2, 0, 2, 2, 0, 0, 2, 1, 0, 2, 0, 1, 2, 2, 0])
    pre = model.predict(df.values[:, :-1])
    df['pre'] = pre
    df_json = df.to_json()
    rsp = make_response(df_json)
    rsp.mimetype = 'application/json'
    return rsp


@app.route('/score', methods=['POST'])
# 作用:模型评估
# 传参：
# 返回值：json ，评价指标
def score():
    data = json.loads(request.get_data())
    df = pd.DataFrame(data)
    pre = df.values[:, -1]
    real = df.values[:, -2]
    c_m = confusion_matrix(real, pre)
    c_m2 = json.dumps(c_m.tolist())
    a_s = accuracy_score(real, pre)
    dict_df = {'score': a_s, 'matrix': c_m2}
    return jsonify(dict_df)

    rsp = make_response(c_m2)
    rsp.mimetype = 'application/json'
    return rsp


if __name__ == '__main__':
    # 判断参数个数
    # knn(db,3)

    # mm='{"user":"root","psw":"bigdata@analysis","host":"10.110.10.15","port":"3306","db_name":"test","table":"iris"}'
    # knn(mm,5)
    #   app.run(debug=True,host='10.72.160.172')
    app.run(debug=True, host='0.0.0.0')