TURN_OVER_COL = "Turnover"
ACC_TURN_OVER_COL = "AccTurnover"
VOLUME_COL = "Volume"
ACC_VOLUME_COL = "AccVolume"
ITEM_COL = "InstrumentID"
SIGNAL_COL = "SIGNAL"
ASSIST_COL = "ASSIST"
ASSIST1_COL = "ASSIST1"
ASSIST2_COL = "ASSIST2"
DATE_COL = "Date"
UPDATE_TIME_COL = "UpdateTime"
LAST_PRICE_COL = "LastPrice"
AVG_PRICE_COL = "AvgPrice"
ACC_AVG_PRICE_COL = "AccAvgPrice"

BUY_PRICE_COL = "BuyPrice"
SELL_PRICE_COL = "SellPrice"
AVG_AD_COL = "AVG_AD"
AVG_BD_COL = "AVG_BD"
MEAN_DEFENSE_COL = "MEAN_DEFENSE"
MEAN_VOLUME_COL = "MEAN_VOLUME"


ASK_PRICE1_COL = "AskPrice1"
BID_PRICE1_COL = "BidPrice1"
ASK_PRICE2_COL = "AskPrice2"
BID_PRICE2_COL = "BidPrice2"
ASK_PRICE3_COL = "AskPrice3"
BID_PRICE3_COL = "BidPrice3"
ASK_PRICE4_COL = "AskPrice4"
BID_PRICE4_COL = "BidPrice4"
ASK_PRICE5_COL = "AskPrice5"
BID_PRICE5_COL = "BidPrice5"

ASK_VOLUME1_COL = "AskVolume1"
BID_VOLUME1_COL = "BidVolume1"
ASK_VOLUME2_COL = "AskVolume2"
BID_VOLUME2_COL = "BidVolume2"
ASK_VOLUME3_COL = "AskVolume3"
BID_VOLUME3_COL = "BidVolume3"
ASK_VOLUME4_COL = "AskVolume4"
BID_VOLUME4_COL = "BidVolume4"
ASK_VOLUME5_COL = "AskVolume5"
BID_VOLUME5_COL = "BidVolume5"

HIGHEST_PRICE_COL = "highest"
LOWEST_PRICE_COL = "lowest"
OPEN_PRICE_COL = "OpenPrice"
HIGH_PRICE_COL = "HighPrice"
LOW_PRICE_COL = "LowPrice"
CLOSE_PRICE_COL = "ClosePrice"
ACTIVE_FACTOR_COL = "ActiveFactor"

CLOSE_COL = "CLOSE"
HIGH_COL = "HIGH"
LOW_COL = "LOW"
ASK_VIB_COL = "ASK_VIB"
BID_VIB_COL = "BID_VIB"
DIFF_AB_COL = "DIFF_AB"
ASK_VIB_NXT_COL = "ASK_VIB_NXT"
BID_VIB_NXT_COL = "BID_VIB_NXT"
DIFF_AB_NXT_COL = "DIFF_AB_NXT"
FEE_COL = "FEE"
SPECIAL_TIME_COL = "SPECIAL_TIME"
MEAN_DEVIATION_COL = "MEAN_DEVIATION"
GT_MIN_RATE_COL = "GT_MIN_RATE"
LT_MAX_RATE_COL = "LT_MAX_RATE"
PRICE_VIB_COL = "PRICE_VIB"
DIFF_PRICE_COL = "DIFF_PRICE"
TREND_COL = "TREND"

TRAN_FEE_PER_LOT = {
    #sqs
    'ag': {'ratio': 0.0001},
    'al': {'value': 6},
    'au': {'value': 10},
    'bu': {'ratio': 0.0002},
    'cu': {'ratio': 0.00015},
    'fu': {'ratio': 0.00005},
    'hc': {'ratio': 0.0002},
    'ni': {'value': 60},
    'pb': {'ratio': 0.00004},
    'rb': {'ratio': 0.0002},
    'ru': {'value': 3},
    'sn': {'value': 6},
    'sp': {'ratio': 0.00005},
    'ss': {'value': 2},
    'wr': {'ratio': 0.00004},
    'zn': {'value': 3},
    'sc': {'value': 20},
    'nr': {'ratio': 0.00002},
    'lu': {'ratio': 0.00002},
    'bc': {'ratio': 0.00001},

    #dss
    'a': {'value': 4},
    'b': {'value': 2},
    'bb': {'ratio': 0.0002},
    'c': {'value': 2.4},
    'cs': {'value': 3},
    'eb': {'value': 6},
    'eg': {'value': 6},
    'fb': {'ratio': 0.0002},
    'i': {'ratio': 0.0002},
    'j': {'ratio': 0.00028},
    'jd': {'ratio': 0.0003},
    'jm': {'ratio': 0.00028},
    'l': {'value': 2},
    'lh': {'ratio': 0.0008},
    'm': {'value': 3},
    'p': {'value': 5},
    'pg': {'value': 12},
    'pp': {'value': 2},
    'rr': {'value': 2},
    'v': {'value': 2},
    'y': {'value': 5},

    # zss
    'SA': {'value': 7},
    'TA': {'value': 3},
    'OI': {'value': 4},
    'FG': {'value': 12},
    'MA': {'value': 8},
    'CF': {'value': 4.3},
    'AP': {'value': 25},
    'RM': {'value': 3},
    'SR': {'value': 3},
    'SF': {'value': 3},
    'UR': {'value': 10},
    'PF': {'value': 6},
    'SM': {'value': 3},
    'PK': {'value': 8},
    'CJ': {'value': 6},
    'CY': {'value': 4},
    'RS': {'value': 4},
    'JR': {'value': 6},
    'LR': {'value': 6},
    'PM': {'value': 60},
    'RI': {'value': 5},
    'ZC': {'value': 300},
    'WH': {'value': 60},

    # zjs
    'IC': {'ratio': 0.000046},
    'IF': {'ratio': 0.000046},
    'IH': {'ratio': 0.000046},
    'IM': {'ratio': 0.000046},
    'T': {'value': 3},
    'TS': {'value': 3},
    'TF': {'value': 3},
}

MIN_DIV = {
    # sqs
    'ag': 1,
    'al': 5,
    'au': 0.02,
    'bu': 1,
    'cu': 10,
    'fu': 1,
    'hc': 1,
    'ni': 10,
    'pb': 5,
    'rb': 1,
    'ru': 5,
    'sn': 10,
    'sp': 2,
    'ss': 5,
    'wr': 1,
    'zn': 5,
    'lu': 1,
    'nr': 5,
    'sc': 0.1,
    'bc': 10,

    #dss
    'a': 1,
    'b': 1,
    'bb': 0.05,
    'c': 1,
    'cs': 1,
    'eb': 1,
    'eg': 1,
    'fb': 0.5,
    'i': 0.5,
    'j': 0.5,
    'jd': 1,
    'jm': 0.5,
    'l': 1,
    'lh': 5,
    'm': 1,
    'p': 2,
    'pg': 1,
    'pp': 1,
    'rr': 1,
    'v': 1,
    'y': 2,

    #zss
    'AP': 1,
    'CF': 5,
    'CJ': 5,
    'CY': 5,
    'FG': 1,
    'JR': 1,
    'LR': 1,
    'MA': 1,
    'OI': 1,
    'PF': 2,
    'PK': 2,
    'PM': 1,
    'RI': 1,
    'RM': 1,
    'RS': 1,
    'SA': 1,
    'SF': 2,
    'SM': 2,
    'SR': 1,
    'TA': 2,
    'UR': 1,
    'WH': 1,
    'ZC': 0.2,

    # zjs
    'IH': 0.2,
    'IF': 0.2,
    'IC': 0.2,
    'IM': 0.2,
    'TS': 0.005,
    'TF': 0.005,
    'T': 0.005,
}

NUM_PER_LOT = {
    #sqs
    "cu": 5,
    "bc": 5,
    "al": 5,
    "zn": 5,
    "pb": 5,
    "ni": 1,
    "sn": 1,
    "au": 1000,
    "ag": 15,
    "rb": 10,
    "wr": 10,
    "hc": 10,
    "ss": 5,
    "sc": 1000,
    "lu": 10,
    "fu": 10,
    "bu": 10,
    "ru": 10,
    "nr": 10,
    "sp": 10,

    #dss
    'a': 10,
    'b': 10,
    'bb': 500,
    'c': 10,
    'cs': 10,
    'eb': 5,
    'eg': 10,
    'fb': 10,
    'i': 100,
    'j': 100,
    'jd': 10,
    'jm': 60,
    'l': 5,
    'lh': 16,
    'm': 10,
    'p': 10,
    'pg': 20,
    'pp': 5,
    'rr': 10,
    'v': 5,
    'y': 10,

    #zss
    'AP': 10,
    'CF': 5,
    'CJ': 5,
    'CY': 5,
    'FG': 20,
    'JR': 20,
    'LR': 20,
    'MA': 10,
    'OI': 10,
    'PF': 5,
    'PK': 5,
    'PM': 50,
    'RI': 20,
    'RM': 10,
    'RS': 10,
    'SA': 20,
    'SF': 5,
    'SM': 5,
    'SR': 10,
    'TA': 5,
    'UR': 20,
    'WH': 20,
    'ZC': 100,

    # zjs
    'IH': 300,
    'IF': 300,
    'IC': 200,
    'IM': 200,
    'TS': 20000,
    'TF': 10000,
    'T': 10000,
}
