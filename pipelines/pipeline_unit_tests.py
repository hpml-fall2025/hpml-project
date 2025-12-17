import datetime as dt

from news import NewsPipeline
from volatility import VolatilityPipeline
from weighting import DynamicWeighting


def test_news_pipeline_hourly():
    # Configure windows via constructor (NOT predict_news_vol kwargs)
    news = NewsPipeline(
        use_gpu=True,
        short_window_hours=6,
        medium_window_hours=30,
        long_window_hours=120,
    )

    # pick an hour that should exist (market hours). 10am ET is usually safe.
    t = dt.datetime(2021, 2, 11, 10, 0, 0)

    n_vol, n_cnt = news.predict_news_vol(
        t,
        k=25,
        feature_weights=(0.5, 0.3, 0.15, 0.05),
        delay_hours=1,
    )

    assert isinstance(n_vol, float), "news vol should be float"
    assert isinstance(n_cnt, int), "headline count should be int"
    assert n_vol >= 0.0, "news vol should be nonnegative"
    assert n_cnt >= 0, "headline count should be nonnegative"

    print()
    print("NEWS OK:", t, "predicted vol=", n_vol, "num headlines used=", n_cnt)
    print()


def test_har_pipeline_hourly():
    har = VolatilityPipeline()

    t1 = dt.datetime(2021, 2, 10, 10, 0, 0)
    t2 = dt.datetime(2021, 2, 11, 10, 0, 0)

    h_pred, true_prev_rv = har.predict_har_vol(t1)

    assert isinstance(h_pred, float), "har prediction should be float"
    assert isinstance(true_prev_rv, float), "true prev RV should be float"
    assert h_pred >= 0.0, "har prediction should be nonnegative"
    assert true_prev_rv >= 0.0, "true prev RV should be nonnegative"

    print()
    print("HAR OK:", t1, "pred=", h_pred, "true_RV_prev_hour=", true_prev_rv)
    print()

    h_pred2, true_prev_rv2 = har.predict_har_vol(t2)
    assert isinstance(h_pred2, float)
    assert isinstance(true_prev_rv2, float)

    print()
    print("HAR OK:", t2, "pred=", h_pred2, "true_RV_prev_hour=", true_prev_rv2)
    print()


def test_dynamic_weighting_two_steps_hourly():
    # IMPORTANT: pass in a NewsPipeline configured with your chosen windows
    har = VolatilityPipeline()
    news = NewsPipeline(
        use_gpu=True,
        short_window_hours=6,
        medium_window_hours=30,
        long_window_hours=120,
    )

    dw = DynamicWeighting(
        lambda_har=0.2,
        lambda_news=0.2,
        warmup_steps=1,
        har_pipe=har,
        news_pipe=news,
        news_feature_weights=(0.5, 0.3, 0.15, 0.05),
        news_delay_hours=1,
        news_k=25,
        norm_window=10,
    )

    t1 = dt.datetime(2021, 2, 10, 10, 0, 0)
    t2 = dt.datetime(2021, 2, 10, 11, 0, 0)

    v1 = dw.predict_weighted_vol(t1)
    assert isinstance(v1, float)
    assert v1 >= 0.0
    print()
    print("DW OK step1:", t1, "V=", v1)
    print()

    v2 = dw.predict_weighted_vol(t2)
    assert isinstance(v2, float)
    assert v2 >= 0.0
    print()
    print("DW OK step2:", t2, "V=", v2)
    print()

    assert dw.rolling_har_error >= 0.0
    assert dw.rolling_news_error >= 0.0


def main():
    test_har_pipeline_hourly()
    test_news_pipeline_hourly()
    test_dynamic_weighting_two_steps_hourly()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()