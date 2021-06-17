from pyspark.sql import DataFrame

def pipe(self, func, *args, **kwargs):
    return func(self, *args, **kwargs)

DataFrame.pipe = pipe
