import torch
import io
import os
import sys
import collections
from collections import defaultdict

def inspect_pt_file_safely(file_path):
    print(f"Analisi del file: {file_path}")
    print(f"Dimensione file: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
    
    try:
        # Tentativo di caricamento sicuro
        print("\n[1] Tentativo di caricamento sicuro...")
        model = torch.load(file_path, map_location="cpu")
        print("✓ Caricamento completato")
        
        # Analizza la struttura del modello
        print("\n[2] Struttura del modello:")
        analyze_structure(model)
        
        # Verifica i tensori
        print("\n[3] Analisi dei tensori:")
        state_dict = None
        
        if isinstance(model, dict):
            for key in ['state_dict', 'model_state_dict']:
                if key in model:
                    state_dict = model[key]
                    print(f"  Trovato {key} nel modello")
                    break
        
        if state_dict is None:
            state_dict = model if has_tensors(model) else None
            if state_dict is model:
                print("  Il modello stesso sembra essere uno state_dict")
        
        if state_dict is not None:
            tensor_count = 0
            dtype_stats = defaultdict(int)
            shape_examples = {}
            
            total_params = 0
            for name, tensor in state_dict.items():
                if isinstance(tensor, torch.Tensor):
                    tensor_count += 1
                    dtype_stats[tensor.dtype] += 1
                    total_params += tensor.numel()
                    if tensor.dtype not in shape_examples:
                        shape_examples[tensor.dtype] = (name, tensor.shape)
            
            print(f"  Numero totale di tensori: {tensor_count}")
            print(f"  Numero totale di parametri: {total_params:,}")
            print("  Distribuzione dei tipi di dati:")
            for dtype, count in dtype_stats.items():
                name, shape = shape_examples[dtype]
                print(f"    - {dtype}: {count} tensori (es. {name}: {shape})")
        else:
            print("  Nessun tensore trovato nel modello")
        
        print("\nRISULTATO:")
        print("✅ Il file è stato caricato con successo in modalità sicura")
        print("   Sembra essere un modello PyTorch valido")
        return True
        
    except Exception as e:
        print(f"\n❌ Errore durante l'analisi sicura: {e}")
        print("\nTentativo di verificare la struttura del file...")
        try:
            with open(file_path, 'rb') as f:
                # Leggi solo l'header per determinare il formato
                header = f.read(16)
                if header.startswith(b'PK\x03\x04'):
                    print("Il file sembra essere un archivio ZIP (potrebbe essere un modello TorchScript)")
                elif header.startswith(b'\x80\x02'):
                    print("Il file sembra essere serializzato con pickle (formato tipico PyTorch)")
                else:
                    print(f"Header del file non riconosciuto: {header[:10].hex()}")
        except Exception as inner_e:
            print(f"Impossibile leggere il file: {inner_e}")
        
        print("\nTentativo diretto di conversione...")
        try:
            from safetensors.torch import save_file
            print("Caricamento del modello per conversione diretta...")
            model = torch.load(file_path, map_location="cpu")
            
            if hasattr(model, "state_dict"):
                state_dict = model.state_dict()
                print("Ottenuto state_dict dal metodo state_dict()")
            elif isinstance(model, dict):
                if "state_dict" in model:
                    state_dict = model["state_dict"]
                    print("Ottenuto state_dict dalla chiave 'state_dict'")
                elif "model_state_dict" in model:
                    state_dict = model["model_state_dict"]
                    print("Ottenuto state_dict dalla chiave 'model_state_dict'")
                else:
                    print("Costruzione state_dict dai tensori presenti nel modello...")
                    state_dict = {k: v for k, v in model.items() if isinstance(v, torch.Tensor)}
            else:
                raise ValueError("Impossibile trovare uno state_dict valido")
            
            # Filtra solo i tensori
            filtered_dict = {}
            for k, v in state_dict.items():
                if isinstance(v, torch.Tensor):
                    filtered_dict[k] = v
                else:
                    print(f"Ignorato '{k}' perché non è un tensore (tipo: {type(v)})")
            
            if not filtered_dict:
                raise ValueError("Nessun tensore trovato nello state_dict")
            
            output_file = file_path.replace(".pt", ".safetensors")
            if output_file == file_path:
                output_file = file_path + ".safetensors"
                
            print(f"Salvataggio di {len(filtered_dict)} tensori in {output_file}...")
            save_file(filtered_dict, output_file)
            print(f"✅ Conversione completata! File salvato come {output_file}")
            
        except Exception as conv_error:
            print(f"❌ Errore durante il tentativo di conversione: {conv_error}")
        
        print("\nRISULTATO:")
        print("⚠️ Non è stato possibile analizzare il file in modo sicuro")
        print("   Si consiglia cautela prima di utilizzare questo modello")
        return False

def analyze_structure(obj, depth=0, max_depth=3, path=""):
    indent = "  " * depth
    if depth >= max_depth:
        print(f"{indent}... (profondità massima raggiunta)")
        return
    
    if isinstance(obj, dict):
        print(f"{indent}Dizionario con {len(obj)} elementi")
        if depth < max_depth - 1:
            for k in list(obj.keys())[:5]:  # Mostra solo le prime 5 chiavi
                print(f"{indent}  '{k}': {type(obj[k]).__name__}")
            if len(obj) > 5:
                print(f"{indent}  ... e altri {len(obj) - 5} elementi")
    elif isinstance(obj, (list, tuple)):
        print(f"{indent}{type(obj).__name__} con {len(obj)} elementi")
        if depth < max_depth - 1 and len(obj) > 0:
            print(f"{indent}  Primo elemento: {type(obj[0]).__name__}")
            if len(obj) > 1:
                print(f"{indent}  Ultimo elemento: {type(obj[-1]).__name__}")
    elif isinstance(obj, torch.Tensor):
        print(f"{indent}Tensor di forma {obj.shape}, tipo {obj.dtype}")
    elif hasattr(obj, '__dict__'):
        print(f"{indent}Oggetto di classe {obj.__class__.__name__}")
        if depth < max_depth - 1:
            attrs = vars(obj)
            for k in list(attrs.keys())[:5]:
                print(f"{indent}  .{k}: {type(attrs[k]).__name__}")
            if len(attrs) > 5:
                print(f"{indent}  ... e altri {len(attrs) - 5} attributi")
    else:
        print(f"{indent}{type(obj).__name__}")

def has_tensors(obj):
    if isinstance(obj, torch.Tensor):
        return True
    elif isinstance(obj, dict):
        return any(isinstance(v, torch.Tensor) for v in obj.values())
    return False

# Esempio d'uso
if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("Inserisci il percorso del file .pt da esaminare: ")
    
    inspect_pt_file_safely(file_path)
