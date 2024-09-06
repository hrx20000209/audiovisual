import time


def get_time(init_time):
    return round((time.time() - init_time) * 1000, 4)
