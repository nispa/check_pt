Controllo se un modello .pt contiene eseguibili.
Il motivo è banale:

se c'è un eseguibile nel modello, 

il modello stesso non è sicuro, 

e conviene convertirlo in .safetensor.


```
python inspect_pt_file.py cfm_model.pt cfm_model.pt
```
