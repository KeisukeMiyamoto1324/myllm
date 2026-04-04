import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

import lightning as L


token_to_id = {
    "what": 0,
    "is": 1,
    "statquest": 2,
    "awesome": 3,
    "<EOS>": 4
}
# {0: 'what', 1: 'is', 2: 'statquest', 3: 'awesome', 4: '<EOS>'}
id_to_token = dict(map(reversed, token_to_id.items()))  

# tensor([[0, 1, 2, 4, 3],
#         [2, 1, 0, 4, 3]])
inputs = torch.tensor(
    [
        [
            token_to_id["what"],
            token_to_id["is"],
            token_to_id["statquest"],
            token_to_id["<EOS>"],
            token_to_id["awesome"],
        ],
        [
            token_to_id["statquest"],
            token_to_id["is"],
            token_to_id["what"],
            token_to_id["<EOS>"],
            token_to_id["awesome"],
        ],
    ]
)

labels = torch.tensor(
    [
        [
            token_to_id["is"],
            token_to_id["statquest"],
            token_to_id["<EOS>"],
            token_to_id["awesome"],
            token_to_id["<EOS>"]
        ],
        [
            token_to_id["is"],
            token_to_id["what"],
            token_to_id["<EOS>"],
            token_to_id["awesome"],
            token_to_id["<EOS>"]
        ],
    ]
)

dataset = TensorDataset(input, labels)
dataloader = DataLoader(dataset)


