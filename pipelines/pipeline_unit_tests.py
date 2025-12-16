import datetime as dt

from news import NewsPipeline
from volatility import VolatilityPipeline
from weighting import DynamicWeighting

def test_news_pipeline():
    news = NewsPipeline()
    d = dt.date(2021, 2, 11)

    n_vol, n_cnt = news.predict_news_vol(d)

    assert n_vol >= 0.0, "news vol should be nonnegative"
    assert n_cnt >= 0, "headline count should be nonnegative"

    print()
    print("NEWS OK:", d, "predicted vol=", n_vol, "num headlines=", n_cnt)
    print()


def test_har_pipeline():
    har = VolatilityPipeline()
    d1 = dt.date(2021, 2, 10)
    d2 = dt.date(2021, 2, 11)

    h_pred, true_prev_rv = har.predict_har_vol(d1)


    assert isinstance(h_pred, float), "har prediction should be float"
    assert isinstance(true_prev_rv, float), "true prev RV should be float"
    assert h_pred >= 0.0, "har prediction should be nonnegative"
    assert true_prev_rv >= 0.0, "true prev RV should be nonnegative"

    print()
    print("HAR OK:", d1, "pred=", h_pred, "true_RV_prev_day=", true_prev_rv)
    print()

    h_pred, true_prev_rv = har.predict_har_vol(d2)
    print()
    print("HAR OK:", d2, "pred=", h_pred, "true_RV_prev_day=", true_prev_rv)
    print()


def test_dynamic_weighting_two_steps():
    dw = DynamicWeighting()

    # Use two consecutive dates so the second call can update rolling errors
    d1 = dt.date(2021, 2, 10)
    d2 = dt.date(2021, 2, 11)

    # Step 1: get weighted prediction + also compute the "true prev RV" from HAR
    # (your DW class should update rolling errors based on previous-step predictions;
    # this test checks that it can run end-to-end)
    v1 = dw.predict_weighted_vol(d1)
    assert v1 >= 0.0
    print()
    print("DW OK step1:", d1, "V=", v1)
    print()

    # Step 2: should now have enough info to update rolling errors internally
    v2 = dw.predict_weighted_vol(d2)
    assert v2 >= 0.0
    print()
    print("DW OK step2:", d2, "V=", v2)
    print()

    # Optional: sanity check that rolling errors exist and are positive
    assert dw.rolling_har_error >= 0.0
    assert dw.rolling_news_error >= 0.0


def main():
    test_news_pipeline()
    test_har_pipeline()
    test_dynamic_weighting_two_steps()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()