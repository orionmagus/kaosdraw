import requests
from datetime import datetime, date
import shelve
import pandas as pd
import numpy as np
import os
import json
import logging
from lotto.models import LottoDraw
from utils.numbers import NumPool, BallInt
logger = logging.getLogger('default')


GAME_URLS = {
    "powerball": "https://www.nationallottery.co.za/index.php?task=results.getHistoricalData&amp;Itemid=272&amp;option=com_weaver&amp;controller=powerball-history",
    "lotto": "https://www.nationallottery.co.za/index.php?task=results.getHistoricalData&amp;Itemid=265&amp;option=com_weaver&amp;controller=lotto-history"
}
GAME = {
    "game": "POWERBALL",
    "url": "https://www.nationallottery.co.za/index.php?task=results.getHistoricalData&amp;Itemid=272&amp;option=com_weaver&amp;controller=powerball-history",
}

POSTKWARGS = {
    "headers": {
        "accept": "*/*",
        "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
        "cache-control": "no-cache",
        "content-type": "application/x-www-form-urlencoded; charset=UTF-8"
    }
}
POSTurl = GAME["url"]
URLDTFMT = '%d/%m/%y'


def datastr(year, game='LOTTO', **kw):
    return "gameName={gameName}&startDate=01/01/{year}&endDate=31/12/{year}&offset=0&limit=250&isAjax=true&isAjax=true".format(year=year, gameName=game)


def url_from(start, game='LOTTO', **kw):
    end = date.today().strftime(URLDTFMT)
    return "gameName={gameName}&startDate={start}&endDate={end}&offset=0&limit=250&isAjax=true&isAjax=true".format(start=start, end=end, gameName=game)


class LottoResults:
    def __init__(self, **kw):

        self.game = kw.get('game', 'LOTTO')
        self.stub = self.game.lower()
        self.url = GAME_URLS.get(self.game.lower(), )
        # self.st_file = 'data/{}_df.json'.format(self.stub[:5])
        # with open(self.st_file, 'r') as f:
        #     # shelve.open(os.path.join(os.curdir, shfn))
        #     self.store = json.load(f)
        # self.jar = requests.cookies.RequestsCookieJar()
        self.sess = requests.Session()

    def update_data(self, s):
        if not s:
            s = LottoDraw.objects.order_by(
                '-draw_date').values_list('draw_date', flat=True)
        try:
            response = self.sess.post(
                self.url,
                data=url_from(s, **{'game': self.game}),
                # timeout=0.8,
                **POSTKWARGS
            )
            if response.ok:
                result = response.json()
                return result.get('data', [])
            else:
                return []
        except Exception as e:
            print(e)
            # logger.error(e)
            return []


def format_df(df):
    cols = ('drawNumber', 'ball1', 'ball2', 'ball3',
            'ball4', 'ball5', 'ball6', 'bonusBall')
    rn = {
        'drawNumber': 'draw_number', 'bonusBall': 'bonus_ball', 'drawDate': 'draw_date',
    }
    for c in cols:
        df[c] = df[c].astype(int)
    df = df.rename(index=str, columns=rn)
    df.draw_date = df.draw_date.apply(
        lambda x: datetime.strptime(x, '%Y/%m/%d')).dt.strftime('%Y-%m-%d')
    df = df.sort_values('draw_number')
    return df


def get_data():
    lotto = LottoResults()
    r = lotto.update_data(None)
    if r:
        if len(r) > 0:
            df = pd.DataFrame(r)
            return format_df(df)
    return False


def update_data(df):
    df = get_data()
    if df:
        records = df.to_dict(orient='records')
        for row in records:
            c, r = LottoDraw.objects.get_or_create(**row)


def cols2nums(cols=('ball1', 'ball2', 'ball3', 'ball4', 'ball5', 'ball6')):
    def _inner(x):

        # kw = dict(
        #     zip(
        #         cols,
        #         # orient='records'
        #         )
        # )
        # print(kw)
        # print(x)
        x['record'] = json.dumps(x.to_dict(), default=str)
        return x
    return _inner


def as_record(data, cols=('ball1', 'ball2', 'ball3', 'ball4', 'ball5', 'ball6')):
    data['record'] = 0
    fn = cols2nums(cols)
    data = data.apply(fn, axis=1)
    return data.drop(list(cols), axis=1)


def load_data_target(records=False):
    from django_pandas.io import read_frame
    from utils.numbers import as_recs, NumPool
    values = ['draw_number', 'ball1', 'ball2',
              'ball3', 'ball4', 'ball5', 'ball6']
    df_kwargs = dict(fieldnames=values, verbose=False,
                     index_col='pk', coerce_float=False)

    df = read_frame(LottoDraw.objects.all(), **df_kwargs)
    if records:
        df = as_recs(df)
        values = ['record']
        # df.record = df.record.apply(lambda k: NumPool(**json.loads(k)))
    else:
        values = values[1:]
    trgs = df[values]
    trgs = trgs.copy()
    trgs.index = trgs.index - 1
    df = df.merge(trgs, how='left', left_index=True,
                  right_index=True, suffixes=['', '_y'])
    if not records:
        return df.fillna(0).astype(int)
    return df


def load_data(fname="data/lotto_daf.pickle"):
    # data = [[1994, '2020/02/08', 3, 41, 37, 28, 51, 42, 14],
    #         [1995, '2020/02/12', 16, 18, 32, 38, 39, 44, 31]]

    # kw = {
    #     'columns': 'drawNumber,drawDate,ball1,ball2,ball3,ball4,ball5,ball6,bonusBall'.split(','),
    #     'index': [n[0] for n in data]
    # }
    # dfr = pd.DataFrame(data, **kw)
    df = pd.read_pickle(fname)
    return df


if __name__ == '__main__':
    lotto = LottoResults()
    df = lotto.dataframe()
    lotto.close()
