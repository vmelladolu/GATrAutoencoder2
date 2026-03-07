
import torch
import torch.nn as nn
import yaml



def build_batch(batch, use_scalar=False, use_energy=False, use_one_hot=False, use_log=False, z_norm=False, stats=None):
        """
        Convierte un Batch de PyG a los tensores que espera el modelo.

        Asume que:
            - batch.pos = (N, 3)
            - batch.x = (N, 6) con [i, j, k, thr1, thr2, thr3]
            - batch.batch = (N,) índices de evento
        """

        # Posiciones                
        mv_v_part = batch.pos
     
                
        N = batch.x.shape[0]
        device = batch.x.device
        
        # Si se va a usar la profundidad
        if use_scalar:
          mv_s_part = batch.k
        else:
          mv_s_part = torch.zeros((N, 1), dtype=torch.float32).to(device) # Placeholder para la parte escalar (profundidad)
        
        # Si los thresholds son (1, 2, 3) o si se quiere usar one-hot encoding [thr1, thr2, thr3]
        if use_one_hot:
            scalars = torch.cat([batch.thr1, batch.thr2, batch.thr3], dim=1) # one-hot de thr1, thr2, thr3
        else:
            scalars = batch.thr
     
        if use_energy and hasattr(batch, "energy"):
            logenergy = batch.energy.view(-1, 1)
        else:
            logenergy = None
            
        batch_idx = batch.batch
        return {
                "mv_v_part": mv_v_part,
                "mv_s_part": mv_s_part,
                "scalars": scalars,
                "logenergy": logenergy,
                "batch_idx": batch_idx,
        }