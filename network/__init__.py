from network.hybird.foundation_model import Network as HybirdNetwork
from network.hybird.audio import AudioCRNN
from network.hybird.audio_vit import AngleProdict
from network.offline.v1.OfflineNet import SAC_model
from network.offline.v1_3.OfflineNet import SAC_Hybird_model
from network.offline.v2.agent import CQLSAC
from network.offline.v4.cql import CQLSAC_hybrid_LSTM as cql_lstm
from network.offline.v5.cql import CQLSAC_hybrid_LSTM as cql_lstm_attention
