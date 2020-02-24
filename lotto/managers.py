from django_pandas.managers import DataFrameQuerySet
from django_pandas.io import read_frame
from utils.numbers import NumPool


class ResultsQuerySet(DataFrameQuerySet):
    def to_dataframe(self, fieldnames=(), verbose=False, index=None, coerce_float=False):
        return super(ResultsQuerySet, self).to_dataframe(
            fieldnames, verbose, index, coerce_float)

    # def to_pivot_table(self, fieldnames=(), verbose=True, values=None, rows=None, cols=None, aggfunc='mean',
    #                    fill_value=None, margins=False, dropna=True):
    #     return super(ResultsQuerySet, self).to_pivot_table(fieldnames, verbose, values, rows, cols, aggfunc,
    #                                                             fill_value, margins, dropna)

    # def to_timeseries(self, fieldnames=(), verbose=True, index=None, storage='wide', values=None, pivot_columns=None,
    #                   freq=None, coerce_float=False, rs_kwargs=None):
    #     return super(ResultsQuerySet, self).to_timeseries(fieldnames, verbose, index, storage, values,
    #                                                            pivot_columns, freq, coerce_float, rs_kwargs)
    def send_pre(self, records, **kwargs):
        for r in records:
            signals.pre_save.send(sender=self.model or r.__class__, instance=r, raw=False,
                                  update_fields=kwargs.keys())

    def send_post(self, records, created=False, **kwargs):
        for r in records:
            signals.post_save.send(sender=self.model or r.__class__, instance=r, raw=False, created=created,
                                   update_fields=kwargs.keys())

    def update(self, **kwargs):
        # self.send_pre(list(self), **kwargs)
        ret = super(ResultsQuerySet, self).update(**kwargs)
        # self.send_post(list(self), **kwargs)
        return ret

    def bulk_create(self, objs, batch_size=None):
        return super(ResultsQuerySet, self).bulk_create(objs, batch_size)

    def create(self, **kwargs):
        return super(ResultsQuerySet, self).create(**kwargs)

    def _batched_insert(self, objs, fields, batch_size):
        super(ResultsQuerySet, self)._batched_insert(objs, fields, batch_size)

    def _insert(self, objs, fields, return_id=False, raw=False, using=None):
        return super(ResultsQuerySet, self)._insert(objs, fields, return_id, raw, using)

    def _update(self, values):
        return super(ResultsQuerySet, self)._update(values)
