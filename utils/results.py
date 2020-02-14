import requests
from datetime import datetime
import shelve
import pandas as pd
import os
import json
import logging

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


def datastr(year, game='LOTTO', **kw):
    return "gameName={gameName}&startDate=01/01/{year}&endDate=31/12/{year}&offset=0&limit=250&isAjax=true&isAjax=true".format(year=year, gameName=game)


class LottoResults:
    def __init__(self, **kw):

        self.game = kw.get('game', 'LOTTO')
        self.stub = self.game.lower()
        self.url = GAME_URLS.get(self.game.lower(), )
        self.st_file = 'data/{}_df.json'.format(self.stub[:5])
        with open(self.st_file, 'r') as f:
            # shelve.open(os.path.join(os.curdir, shfn))
            self.store = json.load(f)
        # self.jar = requests.cookies.RequestsCookieJar()
        self.sess = requests.Session()

    def get_data(self, year):
        try:
            response = self.sess.post(
                self.url,
                data=datastr(year, **{'game': self.game}),
                # timeout=0.8,
                **POSTKWARGS
            )
            if response.ok:
                result = response.json()
                self.store[str(year)] = result.get('data', [])
                return self.store[str(year)]
            else:
                return
        except Exception as e:
            print(e)
            # logger.error(e)
            return

    def dataframe(self):
        records = []
        for y in range(2014, 2021):
            k = str(y)
            if k not in list(self.store.keys()):
                r = self.get_data(k)
            if self.store[k] is not None:
                records.extend(self.store[k])
        df = pd.DataFrame(records)  # .set_index('drawNumber')
        df['dn'] = df.drawNumber.astype(int)
        df = df.sort_values('dn')
        self.df = df.set_index('dn')
        return self.df

    def close(self):
        self.df.to_json('data/{}_daf.json'.format(self.stub[:5]))
        self.df.to_pickle('data/{}_daf.pickle'.format(self.stub[:5]))
        with open(self.st_file, 'w') as f:
            json.dump(self.store, f)


if __name__ == '__main__':
    lotto = LottoResults()
    df = lotto.dataframe()
    lotto.close()
