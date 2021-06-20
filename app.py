from flask.helpers import make_response
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from flask import Flask,  request, render_template
from flask_bootstrap import Bootstrap
from werkzeug.datastructures import FileStorage

nc = Flask(__name__)

y_pred_submit = pd.DataFrame(
    data=None, index=None, columns=None, dtype=None, copy=False)


@nc.route("/")
def check():
    return render_template('upload.html')


@nc.route('/output', methods=['GET', 'POST'])
def csv_reader():
    csv_data = request.files['file']
    # csvファイルのみ受け付ける
    if isinstance(csv_data, FileStorage) and csv_data.content_type == 'text/csv':
        test_x = pd.read_csv(csv_data)
        train_x = pd.read_csv("./data/train_x.csv")
        train_y = pd.read_csv("./data/train_y.csv")
        train_y = train_y["応募数 合計"]
        x_name = ['職場の様子', '交通費別途支給', '1日7時間以下勤務OK', '短時間勤務OK(1日4h以内)', '駅から徒歩5分以内', '学校・公的機関（官公庁）', '派遣スタッフ活躍中', 'Accessのスキルを活かす', 'フラグオプション選択', '派遣形態', '正社員登用あり', '社員食堂あり',
                  '10時以降出社OK', '休日休暇(祝日)', '残業月10時間未満', 'PCスキル不要', '会社概要　業界コード', '勤務地　都道府県コード', '紹介予定派遣', '給与/交通費　交通費', '給与/交通費　給与下限', 'オフィスが禁煙・分煙', '勤務地　市区町村コード', '残業なし']
        train_x = train_x[x_name]

        X_train_valid, X_test, y_train_valid, y_test = train_test_split(
            train_x, train_y, test_size=0.2, random_state=0)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_valid, y_train_valid, test_size=0.2, random_state=0)

        # 第1段階のモデル作成

        first_model_1 = LinearRegression()  # 線形回帰
        first_model_2 = RandomForestRegressor()  # ランダムフォレスト回帰
        first_model_3 = LGBMRegressor()  # LightGBM

        first_model_1.fit(X_train, y_train)
        first_model_2.fit(X_train, y_train)
        first_model_3.fit(X_train, y_train)

        # スタッキングによる予測

        # 第1段階の予測値(この後、メタモデルの入力に使用)
        first_pred_1 = first_model_1.predict(X_valid)
        first_pred_2 = first_model_2.predict(X_valid)
        first_pred_3 = first_model_3.predict(X_valid)

        # 第1段階の予測値を積み重ねる
        stack_pred = np.column_stack(
            (first_pred_1, first_pred_2, first_pred_3))

        # メタモデルの学習
        meta_model = LinearRegression()
        meta_model.fit(stack_pred, y_valid)

        x_test = test_x[x_name]

        pred_1 = first_model_1.predict(x_test)
        pred_2 = first_model_2.predict(x_test)
        pred_3 = first_model_3.predict(x_test)

        stack_pred = np.column_stack((pred_1, pred_2, pred_3))

        y_pred = meta_model.predict(stack_pred)

        y_pred_submit["お仕事No."] = test_x["お仕事No."]
        y_pred_submit["応募数 合計"] = y_pred

    else:
        raise ValueError('data is not csv')

    return render_template('download.html', title="test data", filename="predict.csv")


@nc.route("/export", methods=['POST'])
def export_action():
    filename = request.form['filename']
    response = make_response(y_pred_submit.to_csv(index=False))
    response.headers['Content-Disposition'] = 'attachment; filename='+filename
    response.mimetype = "text/csv"

    return response


if __name__ == '__main__':
    nc.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

nc = Flask(__name__)

y_pred_submit = pd.DataFrame(
    data=None, index=None, columns=None, dtype=None, copy=False)


@nc.route("/")
def check():
    return render_template('upload.html')


@nc.route('/output', methods=['GET', 'POST'])
def csv_reader():
    csv_data = request.files['file']
    # csvファイルのみ受け付ける
    if isinstance(csv_data, FileStorage) and csv_data.content_type == 'text/csv':
        test_x = pd.read_csv(csv_data)
        train_x = pd.read_csv("./data/train_x.csv")
        train_y = pd.read_csv("./data/train_y.csv")
        train_y = train_y["応募数 合計"]
        x_name = ['職場の様子', '交通費別途支給', '1日7時間以下勤務OK', '短時間勤務OK(1日4h以内)', '駅から徒歩5分以内', '学校・公的機関（官公庁）', '派遣スタッフ活躍中', 'Accessのスキルを活かす', 'フラグオプション選択', '派遣形態', '正社員登用あり', '社員食堂あり',
                  '10時以降出社OK', '休日休暇(祝日)', '残業月10時間未満', 'PCスキル不要', '会社概要　業界コード', '勤務地　都道府県コード', '紹介予定派遣', '給与/交通費　交通費', '給与/交通費　給与下限', 'オフィスが禁煙・分煙', '勤務地　市区町村コード', '残業なし']
        train_x = train_x[x_name]

        X_train_valid, X_test, y_train_valid, y_test = train_test_split(
            train_x, train_y, test_size=0.2, random_state=0)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_valid, y_train_valid, test_size=0.2, random_state=0)

        # 第1段階のモデル作成

        first_model_1 = LinearRegression()  # 線形回帰
        first_model_2 = RandomForestRegressor()  # ランダムフォレスト回帰
        first_model_3 = LGBMRegressor()  # LightGBM

        first_model_1.fit(X_train, y_train)
        first_model_2.fit(X_train, y_train)
        first_model_3.fit(X_train, y_train)

        # スタッキングによる予測

        # 第1段階の予測値(この後、メタモデルの入力に使用)
        first_pred_1 = first_model_1.predict(X_valid)
        first_pred_2 = first_model_2.predict(X_valid)
        first_pred_3 = first_model_3.predict(X_valid)

        # 第1段階の予測値を積み重ねる
        stack_pred = np.column_stack(
            (first_pred_1, first_pred_2, first_pred_3))

        # メタモデルの学習
        meta_model = LinearRegression()
        meta_model.fit(stack_pred, y_valid)

        x_test = test_x[x_name]

        pred_1 = first_model_1.predict(x_test)
        pred_2 = first_model_2.predict(x_test)
        pred_3 = first_model_3.predict(x_test)

        stack_pred = np.column_stack((pred_1, pred_2, pred_3))

        y_pred = meta_model.predict(stack_pred)

        y_pred_submit["お仕事No."] = test_x["お仕事No."]
        y_pred_submit["応募数 合計"] = y_pred

    else:
        raise ValueError('data is not csv')

    return render_template('download.html', title="test data", filename="predict.csv")


@nc.route("/export", methods=['POST'])
def export_action():
    filename = request.form['filename']
    response = make_response(y_pred_submit.to_csv(index=False))
    response.headers['Content-Disposition'] = 'attachment; filename='+filename
    response.mimetype = "text/csv"

    return response


if __name__ == '__main__':
    nc.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
