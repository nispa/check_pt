Controllo se un modello .pt contiene eseguibili.
Il motivo è banale:

se c'è un eseguibile nel modello, 

il modello stesso non è sicuro, 

e conviene convertirlo in .safetensor.

```
python inspect_pt_file.py cfm_model.pt
```

Poi, per convertirlo in safetensor, basterà eseguire questo script:
```python

import torch
from safetensors.torch import save_file

# Carica con map_location per sicurezza
model = torch.load("cfm_model.pt", map_location="cpu")

# Estrai lo state_dict nel modo più sicuro possibile
if hasattr(model, "state_dict"):
    state_dict = model.state_dict()
elif isinstance(model, dict):
    if "state_dict" in model:
        state_dict = model["state_dict"]
    elif "model_state_dict" in model:
        state_dict = model["model_state_dict"]
    else:
        state_dict = {k: v for k, v in model.items() if isinstance(v, torch.Tensor)}
else:
    raise ValueError("Impossibile trovare uno state_dict valido")

# Filtra solo i tensori
state_dict = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}

# Salva in formato safetensors
save_file(state_dict, "cfm_model.safetensors")

```

