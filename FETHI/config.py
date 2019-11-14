ROOT = "/home/jbj/zsl/rq/TAFET/"
DATA_ROOT = ROOT + "data/corpus/"
CUDA = "cuda:0"
LOGGER_PATH = ROOT + 'log.txt'

# ------------corpus------------
CORPUS_ROOT = "/home/jbj/zsl/rq/TAFET/data/corpus_raw/"
EMBEDDING_ROOT = "/home/jiabingjing/zhuhao/data/word2vec/GloVe/glove.840B.300d.txt"

WIKI = "Wiki/"
ONTONOTES = "OntoNotes/"

ALL = 'all.txt'
DEV = 'dev.txt'
TRAIN = 'train.txt'
TEST = 'test.txt'

DIR_SETS = [WIKI, ONTONOTES]
FILE_SETS = [TRAIN, DEV, TEST]


# --------- pre-process ---------

CONTEXT_WINDOW = 8
TYPE_SET_INDEX_FILE = "type_set.txt"
REFINED_EMBEDDING_DICT_PATH = ROOT + "data/refined_dict.pkl"

VOCABULARY_LIST = ROOT + "data/vocabulary_list.pkl"
FEATURES_LIST = ROOT + "data/features_list.pkl"

# ---------- embedding ---------
EMBEDDING_DIM = 300
PAD_INDEX = 0
PAD = "[PAD]"
OOV_INDEX = -1
OOV = "[OOV]"

# Wiki

# ----- Struct Attention ----
STRUCT_ATTEN_NUM = 2
# ----- BA LSTM Encoder -----
BALSTM_E_STATE_SIZE = 100
# ----- Char-LSTM Encoder -----
CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ- '$%/1234567890:;"
CHAR_EMBEDDING_DIM = 300
CHAR_OUT = 200
CHAR_SEQ_PAD_LEN = 50
CHARLSTM_E_STATE_SIZE = 200

# ----- Linear  Encoder -----
LINEAR_IN_SIZE = STRUCT_ATTEN_NUM * BALSTM_E_STATE_SIZE + CHAR_OUT
# ------- Inference ---------
INFER_DIM = 100
# --------- loss ------------
PRED_THRESHOLD = 0
# --------- else ------------
DROPOUT = 0.5





# OntoNotes

# # ----- Struct Attention ----
# STRUCT_ATTEN_NUM = 2
# # ----- BA LSTM Encoder -----
# BALSTM_E_STATE_SIZE = 200
# # ----- Char ----------------
#
# #----- Char-LSTM Encoder -----
# CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ- '$%/1234567890:;"
# CHAR_EMBEDDING_DIM = 350
# CHAR_OUT = 200
# CHAR_SEQ_PAD_LEN = 50
# CHARLSTM_E_STATE_SIZE = 250
# # ----- Linear  Encoder -----
# LINEAR_IN_SIZE = STRUCT_ATTEN_NUM * BALSTM_E_STATE_SIZE + CHAR_OUT
# LINEAR_OUT_SIZE = 100
# # ------- Inference ---------
# INFER_DIM = 200
# # --------- loss ------------
# PRED_THRESHOLD = 0
# # --------- else ------------
# DROPOUT = 0.4



