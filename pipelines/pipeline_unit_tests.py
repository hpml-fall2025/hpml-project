import datetime as dt

from news import NewsPipeline
from volatility import VolatilityPipeline
from weighting import Weighting


def test_news_pipeline_hourly():
    news = NewsPipeline(
        use_gpu=True,
        short_window_hours=6,
        medium_window_hours=30,
        long_window_hours=120,
    )

    t = dt.datetime(2021, 2, 11, 10, 0, 0)

    n_vol, n_cnt = news.predict_news_vol(
        t,
        k=25,
        feature_weights=(0.5, 0.3, 0.15, 0.05),
        delay_hours=1,
    )

    assert isinstance(n_vol, float)
    assert isinstance(n_cnt, int)
    assert n_vol >= 0.0
    assert n_cnt >= 0

    print()
    print("NEWS OK:", t, "predicted vol=", n_vol, "num headlines used=", n_cnt)
    print()


def test_har_pipeline_hourly():
    har = VolatilityPipeline()

    t1 = dt.datetime(2021, 2, 10, 10, 0, 0)
    t2 = dt.datetime(2021, 2, 11, 10, 0, 0)

    h_pred, true_prev_rv = har.predict_har_vol(t1)

    assert isinstance(h_pred, float)
    assert isinstance(true_prev_rv, float)
    assert h_pred >= 0.0
    assert true_prev_rv >= 0.0

    print()
    print("HAR OK:", t1, "pred=", h_pred, "true_RV_prev_hour=", true_prev_rv)
    print()

    h_pred2, true_prev_rv2 = har.predict_har_vol(t2)
    assert isinstance(h_pred2, float)
    assert isinstance(true_prev_rv2, float)

    print()
    print("HAR OK:", t2, "pred=", h_pred2, "true_RV_prev_hour=", true_prev_rv2)
    print()


def test_simple_weighting_two_steps_hourly():
    har = VolatilityPipeline()
    news = NewsPipeline(
        use_gpu=True,
        short_window_hours=6,
        medium_window_hours=30,
        long_window_hours=120,
    )

    sw = Weighting(
        lam=-0.1,
        warmup_steps=1,
        har_pipe=har,
        news_pipe=news,
        feature_weights=(0.5, 0.3, 0.15, 0.05),
        delay_hours=1,
        k=25,
        norm_window=10,
    )

    t1 = dt.datetime(2021, 2, 10, 10, 0, 0)
    t2 = dt.datetime(2021, 2, 10, 11, 0, 0)

    v1 = sw.predict_weighted_vol(t1)
    assert isinstance(v1, float)
    assert v1 >= 0.0
    print()
    print("SW OK step1:", t1, "V=", v1)
    print()

    v2 = sw.predict_weighted_vol(t2)
    assert isinstance(v2, float)
    assert v2 >= 0.0
    print()
    print("SW OK step2:", t2, "V=", v2)
    print()


def main():
    test_har_pipeline_hourly()
    test_news_pipeline_hourly()
    test_simple_weighting_two_steps_hourly()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()